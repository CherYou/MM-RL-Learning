"""Own a verl RayWorkerGroup without connecting to or stopping unrelated jobs."""

import uuid
import os
import tempfile
from collections import OrderedDict
import torch
import ray
from ray.util import remove_placement_group
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

from agentic_rl.models import Policy
from .protocol import pack, pad_for_world, object_array


class WorkerRuntime:
    def __init__(self, config):
        self.config = config
        self.owns_ray = not ray.is_initialized()
        self.world_size = config.get("verl_workers", 1)
        nodes = config.get("verl_nodes", 1)
        if nodes < 1 or self.world_size < 1:
            raise ValueError("verl_workers and verl_nodes must be positive")
        if self.world_size % nodes:
            raise ValueError("verl_workers must be divisible by verl_nodes")
        if nodes > 1 and not config.get("ray_address"):
            raise ValueError("Multi-node runs require an explicit ray_address")
        self.pool = None
        self.group = None
        self.generator = None
        if self.owns_ray:
            cpu = config["device"] == "cpu"
            # Ray's local GCS/dashboard RPCs must bypass an inherited HTTP proxy.
            bypass = {
                "127.0.0.1",
                "localhost",
                *os.environ.get("NO_PROXY", "").split(","),
                *os.environ.get("no_proxy", "").split(","),
            }
            os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(bypass - {""}))
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            env = {
                "OMP_NUM_THREADS": str(config.get("cpu_threads", 2)),
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "disabled",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "VLLM_NO_USAGE_STATS": "1",
            }
            if cpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            ray.init(
                address=config.get("ray_address"),
                **(
                    {}
                    if config.get("ray_address")
                    else {
                        "num_cpus": max(4, self.world_size * 2),
                        "num_gpus": 0 if cpu else self.world_size,
                        "object_store_memory": 128 * 1024**2,
                        "_temp_dir": os.path.join(
                            tempfile.gettempdir(), "arl-verl-" + uuid.uuid4().hex[:8]
                        ),
                        "_node_ip_address": "127.0.0.1",
                        "include_dashboard": False,
                    }
                ),
                runtime_env={"env_vars": env},
                log_to_driver=False,
            )
        if config["device"] == "cpu":
            from .worker import CPUWorker

            worker = CPUWorker
        else:
            from .gpu_worker import GPUWorker

            worker = GPUWorker
        self.pool = RayResourcePool(
            [self.world_size // nodes] * nodes, use_gpu=config["device"] != "cpu", max_colocate_count=1
        )
        self.group = RayWorkerGroup(
            self.pool,
            RayClassWithInitArgs(ray.remote(worker), config),
            device_name=config["device"],
            name_prefix="arl_" + uuid.uuid4().hex[:8],
            ray_wait_register_center_timeout=120,
        )
        try:
            self.evidence = self.group.init_model()
            if config["device"] != "cpu":
                from .gpu_rollout import GPUGenerationManager

                self.generator = GPUGenerationManager(self)
            self.policy = RemotePolicy(self, "actor")
        except BaseException:
            self.close()
            raise

    def remote_policy(self, role):
        return RemotePolicy(self, role)

    def close(self):
        try:
            if self.generator is not None:
                self.generator.close()
        finally:
            if self.group is not None:
                for worker in self.group.workers:
                    ray.kill(worker, no_restart=True)
            if self.pool and self.pool.pgs:
                for pg in self.pool.pgs:
                    remove_placement_group(pg)
            if self.owns_ray:
                ray.shutdown()


class RemotePolicy:
    """Tokenizer plus RPCs: the driver never owns trainable model parameters."""

    def __init__(self, runtime, role):
        self.runtime, self.role = runtime, role
        info = runtime.group.describe(role)[0]
        self.tokenizer, self.model_name = info["tokenizer"], info["model_name"]
        self.device = torch.device("cpu")
        self.version = info["version"]
        self.generations = OrderedDict()

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, text, system="", image_path=None):
        if image_path:
            return self.runtime.group.prepare_prompt(text, system, str(image_path), self.role)[0]
        return Policy.prompt(self, text, system)

    def generate_many(self, requests, with_logprobs=False):
        if self.runtime.generator is not None:
            self.runtime.generator.ensure_rollout()
            results = self.runtime.generator.generate_many(requests)
            self.version = self.runtime.generator.version
            tokens = [r.token_ids for r in results]
            log_probs = [r.log_probs for r in results]
        else:
            tokens, log_probs = self._generate_worker_batch(requests)
        for request, out, lp in zip(requests, tokens, log_probs, strict=True):
            self.generations[tuple(request["ids"] + out)] = (len(request["ids"]), lp)
        while len(self.generations) > self.runtime.config.get("rollout_cache_size", 8192):
            self.generations.popitem(last=False)
        return list(zip(tokens, log_probs, strict=True)) if with_logprobs else tokens

    def _generate_worker_batch(self, requests):
        data = DataProto.from_dict(
            tensors={"index": torch.arange(len(requests))},
            non_tensors={"requests": object_array(requests)},
            meta_info={"model_role": self.role},
        )
        # Generation padding has no training weight; drop its outputs immediately.
        extra = (-len(data)) % self.runtime.world_size
        if extra:
            data = DataProto.concat([data, data.select_idxs([0] * extra)])
        output = self.runtime.group.generate(data)
        versions = output.batch["version"][: len(requests)]
        if len(versions.unique()) != 1:
            raise RuntimeError("Rollout workers have inconsistent policy versions")
        self.version = int(versions[0])
        return (
            output.non_tensor_batch["outputs"][: len(requests)].tolist(),
            output.non_tensor_batch["log_probs"][: len(requests)].tolist(),
        )

    def generate(self, prompt_ids, max_new_tokens, extras=None, greedy=False):
        return self.generate_many(
            [{"ids": prompt_ids, "max_new_tokens": max_new_tokens, "extras": extras or {}, "greedy": greedy}]
        )[0]

    @torch.no_grad()
    def score(self, samples, values=False):
        if self.runtime.generator is not None:
            self.runtime.generator.ensure_train()
        data = pack(samples, self.tokenizer.pad_token_id, meta={"model_role": self.role, "values": values})
        data, _ = pad_for_world(data, self.runtime.world_size)
        output = self.runtime.group.compute(data)
        n = len(samples)
        return (
            output.batch["log_probs"][:n],
            data.batch["response_mask"][:n],
            output.batch["values"][:n] if "values" in output.batch else None,
        )

    def snapshot(self, sample):
        sample.old_logp = [0.0] * (len(sample.tokens) - 1)
        spans = sample.turns or [(sample.prompt_length, len(sample.tokens))]
        for start, end in spans:
            key = tuple(sample.tokens[:end])
            if key not in self.generations:
                raise ValueError(
                    "Snapshot has no exact matching generation; refusing to invent rollout logprobs"
                )
            prompt_length, lp = self.generations[key]
            if prompt_length != start or len(lp) != end - start:
                raise ValueError("Generation token/probability alignment mismatch")
            sample.old_logp[start - 1 : end - 1] = lp
        sample.metadata["policy_version"] = self.version
        sample.validate()
        return sample
