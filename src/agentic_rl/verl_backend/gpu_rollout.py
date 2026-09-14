"""Official verl vLLM/SGLang replicas, weight synchronization and async generation."""

import asyncio
import threading
import uuid
import ray
from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager

from .gpu_config import make_gpu_config


class _ServersOnlyManager(AgentLoopManager):
    async def _init_agent_loop_workers(self):
        # The registered LabAgentLoop already runs environments in controller threads.
        # Retain verl's replica launch and global load balancer without duplicate loops.
        self.agent_loop_workers = []


class GPUGenerationManager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.config = make_gpu_config(runtime.config)
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.event_loop.run_forever, name="verl-async-rollout", daemon=True
        )
        self.thread.start()
        self.lock = threading.RLock()
        self.phase = "train"
        self.version = 0
        self.manager = None
        self.allowed_token_ids = None
        info = runtime.group.describe("actor")[0]
        if info["blocked_output_ids"]:
            blocked = set(info["blocked_output_ids"])
            self.allowed_token_ids = [i for i in range(info["vocab_size"]) if i not in blocked]
        asyncio.run_coroutine_threadsafe(self._initialize(), self.event_loop).result(timeout=900)

    async def _initialize(self):
        self.manager = await _ServersOnlyManager.create(self.config, worker_group=self.runtime.group)
        # New replicas start awake. Match upstream RayPPOTrainer's initial
        # sleep before the first weight synchronization; waking an already
        # mapped vLLM 0.12 allocation can fail in its CUDA memory allocator.
        await asyncio.gather(*[replica.sleep() for replica in self.manager.rollout_replicas])
        self.client = AsyncLLMServerManager(
            self.config,
            list(zip(self.manager.server_addresses, self.manager.server_handles, strict=True)),
            self.manager.global_load_balancer,
        )

    def ensure_rollout(self):
        with self.lock:
            if self.phase != "rollout":
                versions = self.runtime.group.enter_rollout()
                if len(set(versions)) != 1:
                    raise RuntimeError("Actor ranks disagree on the version to synchronize into rollout")
                self.version = versions[0]
                self.phase = "rollout"

    def ensure_train(self):
        with self.lock:
            if self.phase != "train":
                self.runtime.group.enter_train()
                self.phase = "train"

    async def _generate(self, request):
        images = None
        image_path = request.get("extras", {}).get("_image_path")
        if image_path:
            with Image.open(image_path) as file:
                images = [file.convert("RGB")]
        params = {
            "max_tokens": request["max_new_tokens"],
            "temperature": 0.0 if request.get("greedy") else 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": True,
        }
        if self.allowed_token_ids is not None:
            params["allowed_token_ids"] = self.allowed_token_ids
        output = await self.client.generate(
            uuid.uuid4().hex, prompt_ids=request["ids"], sampling_params=params, image_data=images
        )
        if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
            raise RuntimeError("Inference server did not return aligned rollout token probabilities")
        return output

    async def _generate_many(self, requests):
        return await asyncio.gather(*[self._generate(request) for request in requests])

    def generate_many(self, requests):
        return asyncio.run_coroutine_threadsafe(self._generate_many(requests), self.event_loop).result(
            timeout=900
        )

    def close(self):
        try:
            # Terminating the owned servers releases their memory directly;
            # a failed engine cannot service a further sleep RPC here.
            if self.manager is not None:
                actors = [server for replica in self.manager.rollout_replicas for server in replica.servers]
                actors.append(self.manager.global_load_balancer)
                for actor in actors:
                    ray.kill(actor, no_restart=True)
        finally:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            self.thread.join(timeout=10)
