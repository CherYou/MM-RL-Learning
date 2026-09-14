"""FSDP actor/rollout/ref worker using verl's model build, sharding and checkpointing.

No instance of this class is created during CPU validation. The asynchronous
vLLM replicas are owned by GPUGenerationManager, not the deprecated sync API.
"""

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from verl.utils.device import get_device_id
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

from agentic_rl.models import Policy
from agentic_rl.harness import load_probes
from .actor import LabPPOActor
from .capo import PartitionBank
from .gpu_config import make_gpu_config
from .losses import reduce_tokens
from .protocol import pack, to_cpu


class _ModelView:
    def __init__(self, worker):
        self.worker, self.lm, self.tokenizer = worker, worker.actor_module_fsdp, worker.tokenizer
        self.device = torch.device("cuda", get_device_id())

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, text, system="", image_path=None):
        return Policy.prompt(self, text, system, image_path)

    def score(self, samples):
        data = pack(samples, self.tokenizer.pad_token_id)
        result = self.worker.actor.infer(data)
        return result["log_probs"], data.batch["response_mask"], None

    def snapshot(self, sample):
        sample.old_logp = self.score([sample])[0][0, : len(sample.tokens) - 1].tolist()
        return sample


class GPUWorker(ActorRolloutRefWorker):
    def __init__(self, lab_config):
        if lab_config["device"] != "cuda" or torch.version.cuda is None:
            raise ValueError("GPUWorker requires the CUDA environment and an explicit device=cuda config")
        expected_uuid = lab_config.get("expected_gpu_uuid")
        if expected_uuid:
            actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
            if actual_uuid.removeprefix("GPU-") != expected_uuid.removeprefix("GPU-"):
                raise RuntimeError(f"GPU affinity mismatch: expected {expected_uuid}, got {actual_uuid}")
        if lab_config.get("lora_rank", 0) and lab_config["algorithm"] == "harness-rl":
            raise ValueError("Harness CAPO requires full MLP parameters, lora_rank=0")
        self.lab_config = lab_config
        self.full_config = make_gpu_config(lab_config)
        # Role includes an immutable step-0 reference; the teacher can have another checkpoint.
        super().__init__(self.full_config.actor_rollout_ref, role="actor_rollout_ref")
        self.version = 0
        self.teacher_worker = None
        self.critic_worker = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        cfg = self.lab_config
        self.actor = LabPPOActor(
            omega_conf_to_dataclass(self.config.actor), self.actor_module_fsdp, self.actor_optimizer, cfg
        )
        self.reference_actor = LabPPOActor(self.config.ref, self.ref_module_fsdp, None, cfg)
        if cfg["algorithm"] in {"opd", "sar-opd", "idt-opd"}:
            teacher_cfg = copy.deepcopy(self.config)
            teacher_cfg.model.path = cfg["teacher_model"]
            self.teacher_worker = ActorRolloutRefWorker(teacher_cfg, role="ref")
            self.teacher_worker.init_model()
            self.teacher_actor = LabPPOActor(
                self.teacher_worker.config.ref, self.teacher_worker.ref_module_fsdp, None, cfg
            )
        elif cfg["algorithm"] == "opsd":
            self.teacher_actor = self.reference_actor
        self.frozen_checkpoints = {}
        frozen_models = {"reference": (self.ref_module_fsdp, self.tokenizer)}
        if self.teacher_worker is not None:
            frozen_models["teacher"] = (self.teacher_worker.ref_module_fsdp, self.teacher_worker.tokenizer)
        for role, (model, tokenizer) in frozen_models.items():
            model.requires_grad_(False)
            self.frozen_checkpoints[role] = FSDPCheckpointManager(
                model=model,
                processing_class=tokenizer,
                checkpoint_config=OmegaConf.create({"save_contents": ["model"], "load_contents": ["model"]}),
            )
            offload_fsdp_model_to_cpu(model)
        if cfg["algorithm"] == "ppo":
            self.critic_worker = CriticWorker(omega_conf_to_dataclass(self.full_config.critic))
            self.critic_worker.init_model()
        if cfg["algorithm"] == "harness-rl" and cfg.get("capo", True):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            view = _ModelView(self)
            bank = PartitionBank(cfg.get("capo_fraction", 0.25))
            bank.probe(self.actor_module_fsdp, load_probes(view, cfg["probe_data"]), view.score)
            self.actor.partitions = bank
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "world_size": self.world_size,
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(),
            "device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "actor_class": "verl.workers.actor.dp_actor.DataParallelPPOActor",
            "sharding": "FSDP1",
            "rollout": self.config.rollout.name,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def describe(self, role="actor"):
        teacher = self.teacher_worker if role == "teacher" and self.teacher_worker is not None else self
        config = self.actor_model_config
        blocked = sorted(
            {
                v
                for key in ("image_token_index", "image_token_id", "video_token_id")
                if (v := getattr(config, key, None)) is not None
            }
        )
        text_config = getattr(config, "text_config", config)
        return {
            "tokenizer": teacher.tokenizer,
            "model_name": teacher.config.model.path,
            "version": self.version,
            "blocked_output_ids": blocked,
            "vocab_size": text_config.vocab_size,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_prompt(self, text, system="", image_path=None, role="actor"):
        if not image_path:
            return _ModelView(self).prompt(text, system)
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]})
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        ids = inputs.pop("input_ids")[0].tolist()
        inputs.pop("attention_mask", None)
        return ids, {**to_cpu(dict(inputs)), "_image_path": str(image_path)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def enter_rollout(self):
        # Ray already runs this actor on its asyncio loop.
        await self.rollout_mode()
        return self.version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def enter_train(self):
        # v0.7.1 removed trainer_mode from the async worker path. The
        # official adapter releases inference weights/KV before actor updates.
        await self.rollout.release()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute(self, data):
        role = data.meta_info.get("model_role", "actor")
        actor = (
            self.actor
            if role == "actor"
            else self.reference_actor
            if role == "reference"
            else self.teacher_actor
        )
        model = actor.actor_module
        load_fsdp_model_to_gpu(model)
        try:
            output = actor.infer(data)
        finally:
            offload_fsdp_model_to_cpu(model)
        if data.meta_info.get("values"):
            if self.critic_worker is None:
                raise ValueError("Value model requested for a non-PPO run")
            values = self.critic_worker.compute_values(data)
            output["values"] = values.batch["values"]
        return DataProto.from_dict(tensors=output)

    def _update_critic(self, data):
        worker = self.critic_worker
        load_fsdp_model_to_gpu(worker.critic_module)
        load_fsdp_optimizer(worker.critic_optimizer, device_id=get_device_id())
        cfg = {**self.lab_config, **data.meta_info["loss_config"]}
        total = 0.0
        for _ in range(cfg.get("ppo_epochs", 1)):
            worker.critic_optimizer.zero_grad(set_to_none=True)
            worker.critic_module.eval()
            for micro in data.split(cfg.get("micro_batch_size", 1)):
                micro = micro.to("cuda")
                inputs = dict(micro.batch)
                values = worker.critic._forward_micro_batch(inputs)
                clipped = inputs["old_values"] + (values - inputs["old_values"]).clamp(
                    -cfg.get("value_clip", 0.2), cfg.get("value_clip", 0.2)
                )
                loss = 0.5 * reduce_tokens(
                    torch.maximum(
                        (values - inputs["returns"]).square(), (clipped - inputs["returns"]).square()
                    ),
                    inputs["response_mask"],
                    cfg,
                    kind="token",
                )
                loss.backward()
                total += float(loss.detach())
            norm = worker.critic._optimizer_step()
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite verl critic gradient")
        worker.critic_lr_scheduler.step()
        offload_fsdp_model_to_cpu(worker.critic_module)
        offload_fsdp_optimizer(worker.critic_optimizer)
        return {"loss/value": total / cfg.get("ppo_epochs", 1), "critic/grad_norm": float(norm)}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update(self, data):
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        load_fsdp_optimizer(self.actor_optimizer, device_id=get_device_id())
        stats = self.actor.update_policy(data)
        self.actor_lr_scheduler.step()
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        offload_fsdp_optimizer(self.actor_optimizer)
        if self.critic_worker is not None:
            stats.update(self._update_critic(data))
        self.version += 1
        stats["verl/policy_version"] = self.version
        return DataProto(meta_info={"metrics": {k: [float(v)] for k, v in stats.items()}})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save(self, directory):
        path = Path(directory)
        self.save_checkpoint(str(path), global_step=self.version)
        if self.critic_worker is not None:
            self.critic_worker.save_checkpoint(str(path / "critic"), global_step=self.version)
        for role, manager in self.frozen_checkpoints.items():
            load_fsdp_model_to_gpu(manager.model)
            manager.save_checkpoint(str(path / role), global_step=self.version)
            offload_fsdp_model_to_cpu(manager.model)
        state = {
            "version": self.version,
            "partitions": self.actor.partitions.state_dict() if self.actor.partitions else None,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        torch.save(state, path / f"worker-rank-{self.rank}.pt")
        if self.rank == 0:
            if not (path / "model").exists():
                (path / "model").symlink_to("huggingface", target_is_directory=True)
            (path / "policy.json").write_text(json.dumps({"model_name": self.config.model.path}) + "\n")
            (path / "frozen-models.json").write_text(
                json.dumps(
                    {
                        "reference": self.config.model.path,
                        "teacher": self.lab_config.get("teacher_model"),
                        "verl_config": OmegaConf.to_container(self.full_config, resolve=True),
                    },
                    indent=2,
                )
                + "\n"
            )
        dist.barrier()
        return {"rank": self.rank, "version": self.version}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore(self, directory):
        path = Path(directory)
        self.load_checkpoint(str(path))
        if self.critic_worker is not None:
            self.critic_worker.load_checkpoint(str(path / "critic"))
        for role, manager in self.frozen_checkpoints.items():
            load_fsdp_model_to_gpu(manager.model)
            manager.load_checkpoint(str(path / role), del_local_after_load=False)
            offload_fsdp_model_to_cpu(manager.model)
        state = torch.load(path / f"worker-rank-{self.rank}.pt", weights_only=False, map_location="cpu")
        self.version = state["version"]
        if self.actor.partitions is not None:
            self.actor.partitions.load_state_dict(state["partitions"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        return self.version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fingerprint(self):
        models = {"actor": self.actor_module_fsdp, "reference": self.ref_module_fsdp}
        if self.teacher_worker is not None:
            models["teacher"] = self.teacher_worker.ref_module_fsdp
        hashes = {}
        for role, model in models.items():
            digest = hashlib.sha256()
            for parameter in model.parameters():
                digest.update(parameter.detach().cpu().float().contiguous().numpy().tobytes())
            hashes[role] = digest.hexdigest()
        return hashes

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
