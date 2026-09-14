"""Single controller for the lab's verl WorkerGroups and algorithm batch builders."""

import json
import os
import platform
import random
import time

import torch
import verl

from agentic_rl.data import ROOT
from agentic_rl.logging import RunLogger
from . import SUPPORTED
from .algorithms import AlgorithmBatchBuilder
from .protocol import pad_for_world
from .runtime import WorkerRuntime


def flatten_numbers(value):
    if isinstance(value, (list, tuple)):
        return [x for part in value for x in flatten_numbers(part)]
    return [float(value)]


def run(config):
    if config["algorithm"] not in SUPPORTED:
        raise ValueError(f"verl supports {sorted(SUPPORTED)}; use TRL for SFT/DPO")
    if verl.__version__ != "0.7.1":
        raise ValueError(f"This integration is pinned to verl 0.7.1, got {verl.__version__}")
    config = {"verl_workers": 1, "micro_batch_size": 1, "ppo_epochs": 1, **config}
    if config["algorithm"] == "sar-opd":
        config.setdefault("medical_steps", max(1, config["steps"] // 2))
    logger = RunLogger(ROOT / config["output"], config)
    runtime = None
    started = time.perf_counter()
    try:
        (logger.path / "status.json").write_text(json.dumps({"status": "running", "backend": "verl"}) + "\n")
        runtime = WorkerRuntime(config)
        builder = AlgorithmBatchBuilder(runtime, config)
        evidence = {
            "verl": verl.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "driver_pid": os.getpid(),
            "host": platform.node(),
            "workers": runtime.evidence,
            "worker_class": "verl.single_controller.ray.RayWorkerGroup",
            "agent_loop": "agentic_rl.verl_backend.agent_loop.LabAgentLoop",
            "initial_parameters": runtime.group.fingerprint(),
        }
        (logger.path / "verl-runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
        start_step = 0
        if config.get("resume"):
            checkpoint = ROOT / config["resume"]
            state = torch.load(checkpoint / "controller.pt", weights_only=False, map_location="cpu")
            if state["algorithm"] != config["algorithm"]:
                raise ValueError("Resume checkpoint algorithm differs from config")
            if state.get("config", {}).get("verl_workers", runtime.world_size) != runtime.world_size:
                raise ValueError("Resume requires the original verl_workers count")
            if state["next_step"] >= config["steps"]:
                raise ValueError("Resume --steps must exceed the saved number of completed steps")
            runtime.group.restore(str(checkpoint))
            builder.store.load_state_dict(state["replay"])
            random.setstate(state["python_rng"])
            start_step = state["next_step"]
            evidence["resume"] = {
                "checkpoint": str(checkpoint),
                "next_step": start_step,
                "restored_parameters": runtime.group.fingerprint(),
            }
            (logger.path / "verl-runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")

        def save(step, path):
            if runtime.generator is not None:
                runtime.generator.ensure_train()
            runtime.group.save(str(path))
            torch.save(
                {
                    "config": config,
                    "next_step": step + 1,
                    "algorithm": config["algorithm"],
                    "replay": builder.store.state_dict(),
                    "python_rng": random.getstate(),
                },
                path / "controller.pt",
            )

        for step in range(start_step, config["steps"]):
            data, samples, metrics, artifacts = builder.build(step)
            if data is None:
                logger.log(step, {**metrics, "update/skipped": 1.0})
                continue
            data, _ = pad_for_world(data, runtime.world_size)
            result = runtime.group.update(data)
            for key, value in result.meta_info["metrics"].items():
                numbers = flatten_numbers(value)
                metrics[key] = sum(numbers) / len(numbers)
            metrics["time/elapsed_seconds"] = time.perf_counter() - started
            logger.log(step, metrics)
            logger.trajectories(step, [s.record() for s in samples])
            for name, artifact in artifacts.items():
                (logger.path / f"{name}-{step:04d}.json").write_text(
                    json.dumps(artifact, ensure_ascii=False) + "\n"
                )
            if (step + 1) % config.get("save_every", 10) == 0:
                save(step, logger.path / f"checkpoint-{step + 1}")
        save(config["steps"] - 1, logger.path / "checkpoint-final")
        evidence["final_parameters"] = runtime.group.fingerprint()
        (logger.path / "verl-runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
        (logger.path / "status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "backend": "verl",
                    "device": config["device"],
                    "steps_requested": config["steps"],
                    "benchmark_validated": False,
                }
            )
            + "\n"
        )
        return logger.path
    except BaseException as error:
        (logger.path / "status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "backend": "verl",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            + "\n"
        )
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            logger.close()
