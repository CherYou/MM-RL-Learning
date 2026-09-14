#!/usr/bin/env python3
"""Audit a real verl GRPO GPU run from its trajectories, parameters and checkpoint."""

import argparse
from collections import defaultdict
import json
import math

from agentic_rl.data import ROOT, read_jsonl
from agentic_rl.rewards import exact_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--report", default="reports/verl-grpo-gpu1-verification.json")
    args = parser.parse_args()
    run = ROOT / args.run
    config = json.loads((run / "config.json").read_text())
    runtime = json.loads((run / "verl-runtime.json").read_text())
    status = json.loads((run / "status.json").read_text())
    metrics = read_jsonl(run / "metrics.jsonl")
    samples = read_jsonl(run / "trajectories.jsonl")
    data = {r["id"]: r for r in read_jsonl(config["dataset"])}
    checks = {}

    def require(name, passed):
        checks[name] = bool(passed)

    require(
        "actual_verl_cuda_run",
        config["backend"] == "verl"
        and config["algorithm"] == "grpo"
        and config["device"] == "cuda"
        and runtime["torch_cuda_build"] is not None,
    )
    require("completed", status["status"] == "completed")
    require(
        "real_data_and_reward",
        config["reward"] == "math"
        and not config.get("smoke_only")
        and "fixtures" not in config["dataset"]
        and config["model"] != "tiny",
    )
    require(
        "physical_gpu_affinity",
        len(runtime["workers"]) == 1
        and runtime["workers"][0]["device_uuid"].removeprefix("GPU-")
        == config["expected_gpu_uuid"].removeprefix("GPU-"),
    )
    require("all_steps_logged", [m["step"] for m in metrics] == list(range(config["steps"])))
    require("finite_metrics", all(math.isfinite(v) for m in metrics for v in m.values()))
    require(
        "nonzero_gradient",
        bool(metrics) and all(m["update/grad_norm"] > 0 and m["update/skipped"] == 0 for m in metrics),
    )
    before, after = runtime["initial_parameters"][0], runtime["final_parameters"][0]
    require("actor_parameters_changed", before["actor"] != after["actor"])
    require("reference_parameters_frozen", before["reference"] == after["reference"])
    require(
        "policy_versions_advance",
        [m["verl/policy_version"] for m in metrics] == list(range(1, config["steps"] + 1)),
    )
    groups = defaultdict(list)
    alignment, rewards_equal, versions_match = True, True, True
    for sample in samples:
        groups[(sample["step"], sample["metadata"]["id"])].append(sample["reward"])
        ids, mask, old = sample["token_ids"], sample["loss_mask"], sample["old_logprobs"]
        alignment &= len(ids) == len(mask) == len(old) + 1 and any(mask)
        alignment &= all(math.isfinite(lp) for lp, keep in zip(old, mask[1:]) if keep)
        for start, end in sample["turn_spans"]:
            alignment &= not any(mask[:start]) and all(mask[start:end])
        row = data[sample["metadata"]["id"]]
        rewards_equal &= sample["reward"] == exact_match(sample["text"], row["answers"])
        versions_match &= sample["metadata"]["policy_version"] == sample["step"]
    require("rollout_count", len(samples) == config["steps"] * config["batch_size"] * config["group_size"])
    require("group_sizes", all(len(g) == config["group_size"] for g in groups.values()))
    require("exact_token_probability_alignment", alignment)
    require("independent_reward_recheck", rewards_equal)
    require("rollout_uses_current_policy_version", versions_match)
    require("nondegenerate_reward_group", any(min(g) < max(g) for g in groups.values()))
    ckpt = run / "checkpoint-final"
    require(
        "model_and_controller_saved",
        (ckpt / "model/config.json").exists()
        and bool(list((ckpt / "model").glob("*.safetensors")))
        and (ckpt / "controller.pt").exists(),
    )
    require(
        "optimizer_and_worker_saved",
        (ckpt / "optim_world_size_1_rank_0.pt").exists() and (ckpt / "worker-rank-0.pt").exists(),
    )
    require("tensorboard_saved", bool(list(run.rglob("events.out.tfevents.*"))))
    evaluation = None
    if (run / "evaluation/summary.json").exists():
        evaluation = json.loads((run / "evaluation/summary.json").read_text())
        predictions = read_jsonl(run / "evaluation/predictions.jsonl")
        held_out = {row["id"]: row for row in read_jsonl(config["eval_dataset"])}
        training_prompts = {row["prompt"] for row in data.values()}
        require(
            "checkpoint_reloaded_on_cuda",
            evaluation["device"] == "cuda"
            and evaluation["rows"] == len(predictions) > 0
            and (ROOT / evaluation["checkpoint"]).resolve() == (ckpt / "model").resolve(),
        )
        require(
            "held_out_rewards_rechecked",
            all(
                p["id"] in held_out
                and held_out[p["id"]]["prompt"] not in training_prompts
                and p["score"] == exact_match(p["text"], held_out[p["id"]]["answers"])
                for p in predictions
            )
            and evaluation["score"] == sum(p["score"] for p in predictions) / len(predictions),
        )
    tokenizer_audit = None
    if (run / "evaluation/tokenizer-audit.json").exists():
        tokenizer_audit = json.loads((run / "evaluation/tokenizer-audit.json").read_text())
        require("checkpoint_tokenizer_preserved", tokenizer_audit["passed"])
    monitor_path = ROOT / "reports" / (run.name + "-monitor.json")
    monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else None
    report = {
        "passed": all(checks.values()),
        "run": str(run),
        "checks": checks,
        "model": config["model"],
        "workers": runtime["workers"],
        "metrics": metrics,
        "rollouts": len(samples),
        "reward_groups": {f"step-{s}/{i}": r for (s, i), r in groups.items()},
        "training_sample_accuracy": sum(x["reward"] for x in samples) / len(samples),
        "runtime": runtime,
        "monitor": monitor,
        "evaluation": evaluation,
        "tokenizer_audit": tokenizer_audit,
        "scope": "Real GPU functional validation on a small training subset; not benchmark accuracy or demonstrated improvement",
    }
    target = ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failed": [k for k, v in checks.items() if not v],
                "report": str(target),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
