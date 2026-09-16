#!/usr/bin/env python3
"""Verify downloaded bytes, data contracts, split isolation and real game assets."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, read_jsonl, question_key


def verify():
    report = {}
    required = [
        "fixtures",
        "gsm8k",
        "dpo",
        "deepmath",
        "opsd",
        "dapo",
        "aime25",
        "medical",
        "medqa",
        "ceval",
        "search",
        "hotpot",
        "geoqa",
        "alfworld-manifests",
        "harness",
    ]
    for name in required:
        folder = ROOT / "data" / name
        meta = json.loads((folder / "manifest.json").read_text())
        for rel, expected in meta["files"].items():
            p = folder / rel
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected["sha256"]:
                raise ValueError(f"Checksum mismatch: {p}")
        sets = {}
        counts = {}
        for file in folder.glob("*.jsonl"):
            rows = read_jsonl(file)
            counts[file.stem] = len(rows)
            if file.stem in {"corpus", "probes"}:
                continue
            keys = set()
            for row in rows:
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError(f"Bad prompt: {file}")
                if not (row.get("answer") or (row.get("chosen") and row.get("rejected"))):
                    raise ValueError(f"Bad target: {file}")
                k = question_key(row["prompt"])
                if k in keys:
                    raise ValueError(f"Duplicate question: {file}")
                keys.add(k)
                if row.get("image"):
                    from PIL import Image

                    with Image.open(ROOT / row["image"]) as image:
                        image.verify()
                if row.get("game_file"):
                    game = ROOT / row["game_file"]
                    if not json.loads(game.read_text()).get("solvable"):
                        raise ValueError(f"Unsolvable game {game}")
                    if not game.with_name("traj_data.json").is_file():
                        raise FileNotFoundError(game)
            sets[file.stem] = keys
        if name != "fixtures":
            train = sets.get("train", set())
            for split, keys in sets.items():
                if split.startswith("eval") and train & keys:
                    raise ValueError(f"Train/eval leakage in {name}")
        report[name] = {
            "verified": True,
            "rows": counts,
            "hashed_files": len(meta["files"]),
            "revision": meta.get("revision"),
        }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/data-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    verify()
