"""CPU worker implementing verl's distributed Worker and actor interfaces.

Used for CPU integration checks and small models. GPU production workers are
provided separately by gpu_worker.py and inherit verl's FSDP implementation.
"""

import os
from pathlib import Path
import random
from datetime import timedelta

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch

from agentic_rl.models import Policy, VisionPolicy, Sample
from agentic_rl.harness import load_probes
from .actor import LabPPOActor
from .capo import PartitionBank
from .protocol import object_array, to_cpu


def actor_config(config):
    return OmegaConf.create(
        {
            "strategy": "fsdp",
            "fsdp_config": {"dtype": "float32"},
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "use_torch_compile": False,
            "ulysses_sequence_parallel_size": 1,
            "entropy_from_logits_with_chunking": False,
            "entropy_checkpointing": False,
            "use_prefix_grouper": False,
            "grad_clip": config.get("max_grad_norm", 1.0),
        }
    )


class CPUWorker(Worker):
    def __init__(self, config):
        super().__init__()
        if config["device"] != "cpu" or torch.version.cuda is not None:
            raise ValueError("CPU worker requires the explicit CPU-only environment")
        self.config = config
        torch.set_num_threads(config.get("cpu_threads", 2))
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
        self.policy = None
        self.models = {}
        self.version = 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        config = self.config
        cls = VisionPolicy if config["algorithm"] == "vision-grpo" else Policy
        self.policy = cls(config["model"], "cpu", config["seed"], config.get("lora_rank", 0))
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=config["learning_rate"],
            weight_decay=0.0,
        )
        self.actor = LabPPOActor(
            actor_config(config), self.policy, self.optimizer, config, local_policy=self.policy
        )
        self.models["actor"] = self.actor
        if config.get("beta", 0) > 0 or config["algorithm"] in {"ppo", "sar-opd", "idt-opd", "opsd"}:
            ref = self.policy.frozen_copy()
            self.models["reference"] = LabPPOActor(actor_config(config), ref, None, config, local_policy=ref)
        if config["algorithm"] in {"opd", "sar-opd", "idt-opd"}:
            teacher = Policy(config["teacher_model"], "cpu", config["seed"] + 1).eval().requires_grad_(False)
            self.models["teacher"] = LabPPOActor(
                actor_config(config), teacher, None, config, local_policy=teacher
            )
        elif config["algorithm"] == "opsd":
            self.models["teacher"] = self.models["reference"]
        if config["algorithm"] == "harness-rl" and config.get("capo", True):
            if config.get("lora_rank", 0):
                raise ValueError("CAPO partitions full MLP parameters; lora_rank must be zero")
            bank = PartitionBank(config.get("capo_fraction", 0.25))
            bank.probe(self.policy, load_probes(self.policy, config["probe_data"]), self.policy.score)
            self.actor.partitions = bank
        torch.manual_seed(config["seed"] + self.rank * 100003)
        random.seed(config["seed"] + self.rank * 100003)
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "world_size": self.world_size,
            "device": "cpu",
            "actor_class": type(self.actor).__mro__[1].__module__ + ".DataParallelPPOActor",
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def describe(self, role="actor"):
        model = self.models[role].local_policy
        return {"tokenizer": model.tokenizer, "model_name": model.model_name, "version": self.version}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_prompt(self, text, system="", image_path=None, role="actor"):
        return to_cpu(self.models[role].local_policy.prompt(text, system, image_path))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate(self, data):
        policy = self.models[data.meta_info.get("model_role", "actor")].local_policy
        outputs = []
        log_probs = []
        for request in data.non_tensor_batch["requests"]:
            out = policy.generate(
                request["ids"], request["max_new_tokens"], request.get("extras"), request.get("greedy", False)
            )
            outputs.append(out)
            sample = Sample(
                request["ids"] + out,
                [0.0] * len(request["ids"]) + [1.0] * len(out),
                len(request["ids"]),
                extras=request.get("extras") or {},
            )
            with torch.no_grad():
                lp, _, _ = policy.score([sample])
            log_probs.append(lp[0, len(request["ids"]) - 1 :].cpu().tolist())
        return DataProto.from_dict(
            tensors={"version": torch.full((len(outputs),), self.version)},
            non_tensors={"outputs": object_array(outputs), "log_probs": object_array(log_probs)},
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute(self, data):
        actor = self.models[data.meta_info.get("model_role", "actor")]
        output = actor.infer(data, values=data.meta_info.get("values", False))
        return DataProto.from_dict(tensors=output)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update(self, data):
        statistics = self.actor.update_policy(data)
        self.version += 1
        statistics["verl/policy_version"] = self.version
        return DataProto(meta_info={"metrics": {k: [float(v)] for k, v in statistics.items()}})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "version": self.version,
            "partitions": self.actor.partitions.state_dict() if self.actor.partitions else None,
        }
        torch.save(state, directory / f"worker-rank-{self.rank}.pt")
        if self.rank == 0:
            self.policy.save(directory)
            for role, model in self.models.items():
                if role != "actor":
                    model.local_policy.save(directory / role)
        dist.barrier()
        return {"rank": self.rank, "version": self.version}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore(self, directory):
        directory = Path(directory)
        for role, actor in self.models.items():
            path = directory if role == "actor" else directory / role
            cls = VisionPolicy if isinstance(actor.local_policy, VisionPolicy) else Policy
            restored = cls(str(path / "model"), "cpu", self.config["seed"], self.config.get("lora_rank", 0))
            actor.local_policy.load_state_dict(restored.state_dict())
            value_file = path / "value_head.pt"
            if value_file.exists():
                actor.local_policy.value_head.load_state_dict(torch.load(value_file, weights_only=True))
        state = torch.load(directory / f"worker-rank-{self.rank}.pt", weights_only=False, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        self.version = state["version"]
        if state["partitions"] is not None:
            self.actor.partitions.load_state_dict(state["partitions"])
        return self.version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def fingerprint(self):
        import hashlib

        def digest(model):
            h = hashlib.sha256()
            for p in model.parameters():
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        return {role: digest(actor.local_policy) for role, actor in self.models.items()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
