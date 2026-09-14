#!/usr/bin/env python3
"""Execute every verl algorithm in actual CPU processes and inspect their checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "grpo": "01-grpo/config.yaml",
    "sar-opd": "02-opd/config.yaml",
    "opd": "02-opd/general-opd/config.yaml",
    "medical-opd": "02-opd/medical-opd.yaml",
    "idt-opd": "02-opd/idt-opd.yaml",
    "search-r1": "03-search-r1/config.yaml",
    "opsd": "04-opsd/config.yaml",
    "retool": "05-retool/config.yaml",
    "dapo": "06-dapo/config.yaml",
    "gspo": "07-gspo/config.yaml",
    "alfworld": "08-alfworld/config.yaml",
    "agentopsd": "09-AgentOPSD/config.yaml",
    "tempo": "09-tempo/config.yaml",
    "vision-grpo": "09-vision-grpo/config.yaml",
    "ppo": "001-ppo/config.yaml",
    "harness-rl": "12-harness-rl/config.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(ENTRIES))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    entries = {k: v for k, v in ENTRIES.items() if not args.only or k in args.only}

    def check(name, config):
        run = f"runs/verl-audit-{stamp}-{name}"
        log = ROOT / f"runs/.logs/verl-audit-{stamp}-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "ACCELERATE_USE_CPU": "true"}
        command = [
            str(ROOT / ".venv-verl/bin/arl"),
            "train",
            config,
            "--backend",
            "verl",
            "--smoke",
            "--verl-workers",
            "2",
            "--output",
            run,
        ]
        with log.open("w") as file:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT, timeout=420
            )
        status_file = ROOT / run / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        result = {
            "command": [".venv-verl/bin/arl", *command[1:]],
            "exit_code": process.returncode,
            "run": run,
            "log": log.relative_to(ROOT).as_posix(),
            "status": status.get("status"),
            "checkpoint": (ROOT / run / "checkpoint-final/model/config.json").exists(),
        }
        result["passed"] = (
            process.returncode == 0 and result["status"] == "completed" and result["checkpoint"]
        )
        return result

    results = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(check, name, config): name for name, config in entries.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as error:
                results[name] = {"passed": False, "error_type": type(error).__name__}
            print(name, "PASS" if results[name]["passed"] else "FAIL", flush=True)
            (ROOT / f"reports/verl-audit-{stamp}.json").write_text(json.dumps(results, indent=2) + "\n")
    latest = ROOT / "reports/verl-audit-latest.json"
    prior = json.loads(latest.read_text()) if args.only and latest.exists() else {}
    latest.write_text(json.dumps({**prior, **results}, indent=2) + "\n")
    raise SystemExit(0 if all(r["passed"] for r in results.values()) else 1)


if __name__ == "__main__":
    main()
