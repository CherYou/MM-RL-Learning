#!/usr/bin/env python3
"""Regenerate LLM entries/configs; retain authored embodied configs in the chapter registry."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

CHAPTERS = [
    ("preliminary", "loss-functions", "fixtures", "native"),
    ("001-ppo", "ppo", "gsm8k", "trl"),
    ("002-dpo", "dpo", "dpo", "trl"),
    ("01-grpo", "grpo", "gsm8k", "trl"),
    ("02-opd", "sar-opd", "medical", "verl"),
    ("02-opd/general-opd", "opd", "deepmath", "verl"),
    ("03-search-r1", "search-r1", "hotpot", "verl"),
    ("04-opsd", "opsd", "opsd", "verl"),
    ("05-retool", "retool", "dapo", "verl"),
    ("06-dapo", "dapo", "dapo", "verl"),
    ("07-gspo", "gspo", "dapo", "trl"),
    ("08-alfworld", "alfworld", "alfworld-manifests", "verl"),
    ("09-AgentOPSD", "agentopsd", "alfworld-manifests", "verl"),
    ("09-tempo", "tempo", "alfworld-manifests", "verl"),
    ("09-vision-grpo", "vision-grpo", "geoqa", "verl"),
    ("12-harness-rl", "harness-rl", "hotpot", "verl"),
]

EMBODIED_CHAPTERS = [
    (f"{number}-{algorithm}", algorithm, "embodied/fetch-reach", "embodied")
    for number, algorithm in [(13, "sac"), (14, "td3"), (15, "her"), (16, "iql")]
]


def write_verl_configs(folder, config, stem="verl"):
    cpu = {**config, "backend": "verl", "device": "cpu", "verl_workers": 2, "micro_batch_size": 1}
    gpu = {
        **cpu,
        "device": "cuda",
        "verl_nodes": 1,
        "rollout_engine": "vllm",
        "rollout_tensor_parallel_size": 1,
        "rollout_gpu_memory_utilization": 0.5,
        "max_context_tokens": 4096,
        "gradient_checkpointing": False,
    }
    (folder / f"{stem}.yaml").write_text(yaml.safe_dump(cpu, sort_keys=False))
    (folder / f"{stem}-gpu.yaml").write_text(yaml.safe_dump(gpu, sort_keys=False))


def main():
    for chapter, algorithm, dataset, backend in CHAPTERS:
        folder = ROOT / chapter
        folder.mkdir(parents=True, exist_ok=True)
        config = {
            "algorithm": algorithm,
            "backend": backend,
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "dataset": f"data/{dataset}/train.jsonl",
            "eval_dataset": f"data/{dataset}/eval.jsonl",
            "steps": 20,
            "batch_size": 1,
            "group_size": 4,
            "max_new_tokens": 256,
            "max_turns": 8,
            "learning_rate": 5e-6,
            "beta": 0.0,
            "seed": 42,
            "cpu_threads": 2,
        }
        if algorithm in {"opd", "sar-opd"}:
            config["teacher_model"] = "Qwen/Qwen2.5-1.5B-Instruct"
        if algorithm == "sar-opd":
            config.update(
                general_dataset="data/ceval/train.jsonl",
                eval_dataset="data/medqa/eval.jsonl",
                teacher_model="medical_sft",
            )
        if algorithm in {"alfworld", "agentopsd", "tempo"}:
            config.update(
                eval_dataset="data/alfworld-manifests/eval-unseen.jsonl",
                environment="alfworld",
                max_turns=50,
                max_new_tokens=96,
            )
        if algorithm == "tempo":
            config.update(macro_horizon=4, critic_samples=4, critic_tokens=128, warmup_steps=2)
        if algorithm == "dapo":
            config.update(
                clip_low=0.2, clip_high=0.28, soft_length=192, mask_truncated=True, max_resample_batches=8
            )
        if algorithm == "gspo":
            config.update(clip_low=0.0003, clip_high=0.0004, loss="grpo")
        if algorithm == "ppo":
            config.update(beta=0.02, ppo_epochs=2, batch_size=4)
        if algorithm == "vision-grpo":
            config["model"] = "Qwen/Qwen2.5-VL-3B-Instruct"
        if algorithm == "harness-rl":
            config.update(
                probe_data="data/harness/probes.jsonl", capo=True, capo_fraction=0.25, process_coef=0.1
            )
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if algorithm not in {"loss-functions", "dpo"}:
            write_verl_configs(folder, config)
        levels = len(Path(chapter).parts)
        pre = f'''#!/usr/bin/env python3
"""{chapter}: local {algorithm} entry point. Run from any directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[{levels}]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.cli import main
'''
        call = (
            'main(["loss-demo"])'
            if algorithm == "loss-functions"
            else f'main(["train", "{chapter}/config.yaml", *sys.argv[1:]])'
        )
        (folder / "train.py").write_text(pre + f'\nif __name__ == "__main__":\n    {call}\n')
        if algorithm != "loss-functions":
            (folder / "eval.py").write_text(
                pre
                + f'\nif __name__ == "__main__":\n    main(["eval", "{chapter}/config.yaml", *sys.argv[1:]])\n'
            )
        (folder / "prepare_data.py").write_text(f'''#!/usr/bin/env python3
"""Prepare this chapter's shared data, using the versioned central downloader."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[{levels}]
if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT/"scripts/prepare_all.py"), *sys.argv[1:]],check=True,cwd=ROOT)
''')
    base = yaml.safe_load((ROOT / "02-opd/config.yaml").read_text())
    for name, changes in [
        ("sft", {"algorithm": "sft", "backend": "trl", "eval_dataset": "data/medical/eval.jsonl"}),
        ("medical-opd", {"algorithm": "opd"}),
        ("idt-opd", {"algorithm": "idt-opd"}),
    ]:
        variant = {**base, **changes}
        (ROOT / f"02-opd/{name}.yaml").write_text(yaml.safe_dump(variant, sort_keys=False))
        if name != "sft":
            write_verl_configs(ROOT / "02-opd", variant, f"{name}-verl")
    existing_path = ROOT / "configs/chapters.json"
    existing_meta = {}
    if existing_path.exists():
        try:
            for row in json.loads(existing_path.read_text(encoding="utf-8")):
                if isinstance(row, dict) and "chapter" in row:
                    existing_meta[row["chapter"]] = row
        except json.JSONDecodeError:
            existing_meta = {}
    registry = []
    for c, a, d, b in CHAPTERS + EMBODIED_CHAPTERS:
        row = {"chapter": c, "algorithm": a, "dataset": d, "backend": b}
        prior = existing_meta.get(c, {})
        for key, value in prior.items():
            if key not in row:
                row[key] = value
        registry.append(row)
    existing_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
