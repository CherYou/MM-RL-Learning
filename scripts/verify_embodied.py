#!/usr/bin/env python3
"""Audit actual saved FetchReach experiments; never infer success from exit status alone."""

import hashlib
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from agentic_rl.embodied.agents import Agent
from agentic_rl.embodied.runner import actor_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence, checks = {}, {}
    for algorithm in ("sac", "td3", "her", "iql"):
        run = ROOT / f"runs/embodied-{algorithm}-default-20260911"
        result = json.loads((run / "validation.json").read_text())
        checkpoint = run / "checkpoint-final/agent.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        agent = Agent(state["config"])
        agent.load_state_dict(state)
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        events = EventAccumulator(str(run / "tensorboard")).Reload()
        tags = events.Tags()["scalars"]
        checks[algorithm + "/checkpoint_hash"] = (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
        )
        checks[algorithm + "/actor_reloaded"] = actor_hash(agent) == result["final_actor_sha256"]
        checks[algorithm + "/actor_changed"] = (
            result["actor_change_l2"] > 0 and result["initial_actor_sha256"] != result["final_actor_sha256"]
        )
        checks[algorithm + "/finite_weights"] = all(
            torch.isfinite(value).all().item()
            for network in state["networks"].values()
            for value in network.values()
        )
        checks[algorithm + "/finite_metrics"] = all(
            math.isfinite(value) for row in metrics for value in row.values()
        )
        checks[algorithm + "/nonzero_updates"] = result["updates"] > 0
        checks[algorithm + "/exact_reload_eval"] = (
            result["reload_exact"] and result["reloaded"] == result["final"]
        )
        checks[algorithm + "/tensorboard"] = all(
            tag in tags for tag in ("loss/actor", "loss/critic", "eval/return", "eval/final_success_rate")
        )
        checks[algorithm + "/actual_physics"] = (
            result["environment"] == "FetchReach-v4"
            and result["device"] == "cpu"
            and result["versions"]["mujoco"] == "3.3.7"
        )
        checks[algorithm + "/training_interactions"] = result["training_environment_steps"] == (
            0 if algorithm == "iql" else 5000
        )
        independent = json.loads((ROOT / f"reports/embodied-{algorithm}-evaluation.json").read_text())
        checks[algorithm + "/independent_eval"] = {row["seed"] for row in independent["episodes"]} == set(
            range(20000, 20010)
        )
        evidence[algorithm] = {
            "run": run.relative_to(ROOT).as_posix(),
            "validation": result,
            "independent_evaluation": independent,
            "tensorboard_tags": tags,
        }
    checks["her/real_and_relabelled_separate"] = (
        evidence["her"]["validation"]["her_transitions"] == 20000
        and evidence["her"]["validation"]["replay_transitions"] == 25000
    )
    dataset = ROOT / "data/embodied/fetch-reach/train.npz"
    meta = json.loads(dataset.with_suffix(".json").read_text())
    checks["iql/dataset_unchanged"] = (
        meta["sha256"]
        == evidence["iql"]["validation"]["dataset"]["sha256"]
        == hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    checks["iql/fixed_2000_transitions"] = (
        evidence["iql"]["validation"]["replay_transitions"] == meta["transitions"] == 2000
    )
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "experiments": evidence,
        "limits": "One training seed, short budgets, low-dimensional state-goal policies; no benchmark or VLA capability claim",
    }
    (ROOT / "reports/embodied-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failed": [key for key, value in checks.items() if not value],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
