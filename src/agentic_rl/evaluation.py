"""Held-out checkpoint evaluation; never labels CPU smoke scores as benchmarks."""

import json
from pathlib import Path
import torch
from .data import ROOT, read_jsonl, report_path, write_jsonl
from .models import Policy, VisionPolicy
from .rollout import single_rollout, agent_rollout
from .harness import harness_rollout
from .trainers import supervised_sample


def evaluate(config, checkpoint, limit=32, output=None):
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    path = Path(checkpoint)
    if not path.is_absolute():
        path = ROOT / path
    if (path / "model").is_dir():
        path = path / "model"
    cls = VisionPolicy if config["algorithm"] == "vision-grpo" else Policy
    policy = cls(str(path), config["device"], config["seed"])
    rows = read_jsonl(config["eval_dataset"])
    train = read_jsonl(config["dataset"])
    if {r["prompt"] for r in rows} & {r["prompt"] for r in train}:
        raise ValueError("Train/eval overlap")
    chosen = rows if limit == 0 else rows[:limit]
    config = {**config, "reward": "math"}  # Evaluation never uses debug rewards.
    records = []
    with torch.no_grad():
        for row in chosen:
            if config["algorithm"] == "dpo":
                samples = [
                    supervised_sample(policy, row, k, config["max_new_tokens"])
                    for k in ("chosen", "rejected")
                ]
                lp, m, _ = policy.score(samples)
                sums = (lp * m).sum(-1)
                records.append(
                    {
                        "id": row["id"],
                        "score": float(sums[0] > sums[1]),
                        "chosen_minus_rejected_logp": float(sums[0] - sums[1]),
                    }
                )
            elif config["algorithm"] == "harness-rl":
                calls, score = harness_rollout(policy, row, config, row["id"])
                records.append({"id": row["id"], "score": score, "calls": [c.as_dict() for c in calls]})
            else:
                sampler = (
                    agent_rollout
                    if config["algorithm"] in {"search-r1", "retool", "alfworld", "tempo", "agentopsd"}
                    else single_rollout
                )
                sample = sampler(policy, row, config, greedy=True)
                # ALFWorld reward is exact environment success, all other tasks exact match.
                records.append({"id": row["id"], "score": sample.reward, **sample.record()})
    out = Path(output) if output else path.parent / "evaluation"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", records)
    summary = {
        "checkpoint": report_path(path),
        "dataset": report_path(config["eval_dataset"]),
        "rows": len(records),
        "score": sum(r["score"] for r in records) / len(records),
        "seed": config["seed"],
        "device": config["device"],
        "metric": "chosen logprob preference accuracy (length sensitive)"
        if config["algorithm"] == "dpo"
        else "task verifier",
        "scope": "CPU smoke only"
        if config.get("smoke_only")
        else "selected held-out learning subset; not paper benchmark reproduction",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary
