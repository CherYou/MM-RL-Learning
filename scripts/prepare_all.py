#!/usr/bin/env python3
"""Idempotent preparation; local manifests are validated by scripts/verify_data.py."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    commands = [["scripts/prepare_data.py", "--dataset", "fixtures"]]
    for key in ["gsm8k", "deepmath", "opsd", "dapo", "aime25", "medical", "search", "geoqa"]:
        if args.full or not (ROOT / f"data/{key}/manifest.json").exists():
            limit = 0 if args.full or key in {"gsm8k", "geoqa", "aime25"} else 1024
            commands.append(["scripts/prepare_data.py", "--dataset", key, "--limit", str(limit)])
    if args.full or not (ROOT / "data/ceval/manifest.json").exists():
        commands.append(["scripts/prepare_ceval.py"])
    if not (ROOT / "data/hotpot/manifest.json").exists():
        commands.append(["scripts/prepare_environments.py", "--only", "retrieval"])
    if not (ROOT / "data/medqa/manifest.json").exists():
        commands.append(["scripts/prepare_environments.py", "--only", "medqa"])
    if not (ROOT / "data/alfworld-manifests/manifest.json").exists():
        commands.append(["scripts/prepare_environments.py", "--only", "alfworld", "--download-alfworld"])
    commands += [["scripts/prepare_fixtures.py"], ["scripts/verify_data.py"]]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
