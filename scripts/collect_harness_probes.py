#!/usr/bin/env python3
"""Collect successful on-policy interface calls for CAPO activation probing."""

import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.cli import load_config
from agentic_rl.data import ROOT, read_jsonl, write_jsonl
from agentic_rl.harness import harness_rollout
from agentic_rl.models import Policy


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--output", default="data/harness/onpolicy-probes.jsonl")
    a = p.parse_args()
    torch.set_num_threads(2)
    c = load_config("12-harness-rl/config.yaml", {"model": a.model})
    policy = Policy(a.model, "cpu", c["seed"])
    probes = []
    for row in read_jsonl(c["dataset"])[: a.limit]:
        for i in range(c["group_size"]):
            calls, score = harness_rollout(policy, row, c, f"{row['id']}-{i}")
            if score == 1.0:
                probes.extend(
                    {
                        **call.as_dict(),
                        "reward": score,
                        "source_model": a.model,
                        "provenance": "successful on-policy rollout",
                    }
                    for call in calls
                )
    if not probes:
        raise SystemExit(
            "No successful rollouts: improve base policy or increase sampling before CAPO probing."
        )
    write_jsonl(ROOT / a.output, probes)
    print("Saved successful calls:", len(probes))
