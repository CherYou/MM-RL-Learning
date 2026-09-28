#!/usr/bin/env python3
"""CPU-only end-to-end entrypoint audit. Each process has a bounded timeout."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    chapters = json.loads((ROOT / "configs/chapters.json").read_text())
    trials = [
        (r["algorithm"], r["chapter"] + "/config.yaml", "native" if r["backend"] == "verl" else r["backend"])
        for r in chapters
        if r["algorithm"] != "loss-functions" and r["backend"] != "embodied"
    ]
    trials += [
        ("sft", "15-medical-opd/sft.yaml", "trl"),
        ("idt-opd", "15-medical-opd/idt-opd.yaml", "native"),
        ("grpo-native", "03-grpo/config.yaml", "native"),
        ("gspo-native", "05-gspo/config.yaml", "native"),
        ("dpo-native", "06-dpo/config.yaml", "native"),
        ("ppo-native", "02-ppo/config.yaml", "native"),
        ("dapo-trl-loss", "04-dapo/config.yaml", "trl"),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "ACCELERATE_USE_CPU": "true",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {}
    for name, cfg, backend in trials:
        if a.only and name not in a.only:
            continue
        output = f"runs/cpu-audit-{stamp}/{name}"
        log = ROOT / f"runs/.logs/cpu-audit-{stamp}-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "agentic_rl.cli",
            "train",
            cfg,
            "--smoke",
            "--backend",
            backend,
            "--output",
            output,
        ]
        started = time.monotonic()
        with log.open("w") as f:
            try:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180
                )
            except subprocess.TimeoutExpired:
                report[name] = {
                    "passed": False,
                    "error": "180s timeout",
                    "log": log.relative_to(ROOT).as_posix(),
                }
                continue
        status = ROOT / output / "status.json"
        success = (
            result.returncode == 0
            and status.exists()
            and json.loads(status.read_text())["status"] == "completed"
        )
        report[name] = {
            "passed": success,
            "exit_code": result.returncode,
            "seconds": round(time.monotonic() - started, 2),
            "run": output,
            "log": log.relative_to(ROOT).as_posix(),
        }
        print(name, report[name], flush=True)
    target = ROOT / f"reports/cpu-audit-{stamp}.json"
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    latest = ROOT / "reports/cpu-audit-latest.json"
    combined = json.loads(latest.read_text()) if a.only and latest.exists() else {}
    combined.update(report)
    latest.write_text(json.dumps(combined, indent=2) + "\n")
    if not all(r["passed"] for r in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
