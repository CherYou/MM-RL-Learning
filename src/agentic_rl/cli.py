"""One local command interface for chapter training, evaluation and diagnostics."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import yaml
from .data import ROOT

DEFAULTS = {
    "model": "tiny",
    "device": "cpu",
    "seed": 42,
    "cpu_threads": 2,
    "steps": 4,
    "batch_size": 1,
    "group_size": 4,
    "max_new_tokens": 32,
    "max_turns": 3,
    "max_context_tokens": 4096,
    "max_prompt_tokens": 1024,
    "learning_rate": 1e-4,
    "beta": 0.0,
    "save_every": 10,
    "backend": "native",
}


def load_config(path, overrides=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    data = yaml.safe_load(path.read_text())
    config = {**DEFAULTS, **data, **(overrides or {})}
    config.setdefault(
        "output",
        f"runs/{config['algorithm']}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000000:06d}",
    )
    if config["steps"] < 1 or config["batch_size"] < 1 or config["group_size"] < 2:
        raise ValueError("Positive steps and batch_size, group_size >= 2 required")
    return config


def smoke_config(config):
    if config["backend"] == "embodied":
        return {
            **config,
            "steps": 300,
            "warmup_steps": 50,
            "batch_size": 32,
            "hidden_dim": 64,
            "eval_episodes": 3,
            "device": "cpu",
        }
    config = {
        **config,
        "model": "tiny",
        "teacher_model": "tiny",
        "lora_rank": 0,
        "device": "cpu",
        "steps": 2,
        "batch_size": 1,
        "group_size": 4,
        "max_new_tokens": 12,
        "max_turns": 2,
        "max_context_tokens": 4096,
        "max_observation_chars": 200,
        "critic_tokens": 8,
        "critic_samples": 2,
        "environment": "toy",
        "reward": "debug_token",
        "mask_truncated": False,
        "soft_length": 9,
        "cpu_threads": 2,
        "max_resample_batches": 3,
        "smoke_only": True,
        "dataset": "data/fixtures/train.jsonl",
        "eval_dataset": "data/fixtures/eval.jsonl",
    }
    algo = config["algorithm"]
    if algo == "dpo":
        config.update(dataset="data/fixtures/dpo-train.jsonl", eval_dataset="data/fixtures/dpo-eval.jsonl")
    elif algo == "vision-grpo":
        config.update(
            dataset="data/fixtures/vision-train.jsonl", eval_dataset="data/fixtures/vision-eval.jsonl"
        )
    elif algo in {"sar-opd", "idt-opd"}:
        # General-domain smoke rows have a third disjoint set, never the eval split.
        config["general_dataset"] = "data/fixtures/general-train.jsonl"
    elif algo == "tempo":
        config.update(steps=3, warmup_steps=1, macro_horizon=1)
    return config


def train(config):
    if config["backend"] == "embodied":
        from .embodied.runner import train as train_embodied

        result = train_embodied(config)
        print(f"Saved run: {result}")
        return result
    if config["backend"] == "verl":
        from .verl_backend import needs_environment_switch, run_verl

        if needs_environment_switch(config):
            return run_verl(config)
        if config.get("resume"):
            import torch

            state = torch.load(
                ROOT / config["resume"] / "controller.pt", map_location="cpu", weights_only=False
            )
            original = state.get("config", {})
            for key in ("model", "teacher_model", "seed", "medical_steps"):
                if key in original:
                    config[key] = original[key]
    if config["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["ACCELERATE_USE_CPU"] = "true"
    else:
        os.environ["ACCELERATE_USE_CPU"] = "false"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if config["algorithm"] in {"opd", "sar-opd", "idt-opd"} and config.get("teacher_model") == "medical_sft":
        teacher = load_config(
            "02-opd/sft.yaml",
            {
                "model": config["model"],
                "device": config["device"],
                "steps": config.get("teacher_sft_steps", config["steps"]),
                "dataset": config.get("teacher_dataset", "data/medical/train.jsonl"),
                "eval_dataset": config.get("teacher_eval_dataset", "data/medical/eval.jsonl"),
                "max_new_tokens": config["max_new_tokens"],
                "output": config["output"] + "-teacher-sft",
            },
        )
        if config.get("smoke_only"):
            teacher["smoke_only"] = True
        teacher_run = train(teacher)
        config = {**config, "teacher_model": str(teacher_run / "checkpoint-final/model")}
    from .trainers import run_native

    if config["backend"] == "verl":
        from .verl_backend import run_verl

        result = run_verl(config)
    elif config["backend"] == "trl":
        from .trl_backend import run_trl

        result = run_trl(config)
    else:
        result = run_native(config)
    print(f"Saved run: {result}")
    return result


def doctor():
    import torch

    packages = [
        "torch",
        "torchvision",
        "transformers",
        "trl",
        "accelerate",
        "peft",
        "datasets",
        "alfworld",
        "textworld",
        "tensorboard",
        "streamlit",
    ]
    result = {
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "torch_cuda_build": torch.version.cuda,
        "default_device": "cpu",
        "pytrio_installed": False,
        "datasets": {},
    }
    try:
        importlib.metadata.version("pytrio")
        result["pytrio_installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass
    for path in sorted((ROOT / "data").glob("*/manifest.json")):
        result["datasets"][path.parent.name] = json.loads(path.read_text()).get("files", {})
    (ROOT / "reports/doctor.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "datasets"}, indent=2))
    print("Dataset manifests:", ", ".join(result["datasets"]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("config")
    t.add_argument("--smoke", action="store_true")
    for flag, typ in [
        ("model", str),
        ("teacher-model", str),
        ("device", str),
        ("steps", int),
        ("backend", str),
        ("output", str),
        ("dataset", str),
        ("eval-dataset", str),
        ("verl-workers", int),
        ("verl-nodes", int),
        ("ray-address", str),
        ("resume", str),
    ]:
        t.add_argument("--" + flag, type=typ)
    e = sub.add_parser("eval")
    e.add_argument("config")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--limit", type=int, default=32)
    e.add_argument("--output")
    e.add_argument("--smoke", action="store_true")
    e.add_argument("--eval-dataset")
    sub.add_parser("doctor")
    sub.add_parser("loss-demo")
    args = p.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "loss-demo":
        from .loss_demo import run

        run()
        return
    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    for k in (
        "model",
        "teacher_model",
        "device",
        "steps",
        "backend",
        "output",
        "dataset",
        "eval_dataset",
        "verl_workers",
        "verl_nodes",
        "ray_address",
        "resume",
    ):
        if getattr(args, k, None) is not None:
            config[k] = getattr(args, k)
    if args.command == "train":
        train(config)
        return
    if config["backend"] == "embodied":
        from .embodied.runner import main as embodied_main

        arguments = ["eval", "--checkpoint", args.checkpoint, "--episodes", str(args.limit)]
        if args.output:
            arguments.extend(["--output", args.output])
        return embodied_main(arguments)
    from .evaluation import evaluate

    evaluate(config, args.checkpoint, args.limit, args.output)


if __name__ == "__main__":
    main()
