#!/usr/bin/env python3
"""Audit actual verl artifacts, configurations and installed environments, without GPU execution."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import yaml

from agentic_rl.verl_backend import SUPPORTED
from smoke_verl import ENTRIES

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            print("FAILED", name)

    latest = read("reports/verl-audit-latest.json")
    resume = read("reports/verl-resume.json")
    require("all_algorithms_have_entries", {"opd" if x == "medical-opd" else x for x in ENTRIES} == SUPPORTED)
    require("all_16_execution_reports_present", set(latest) == set(ENTRIES))
    evidence = {}
    for name, item in latest.items():
        # Prefer the TEMPO execution after the total-turn limit and resume additions.
        path = Path(resume["initial_run"] if name == "tempo" else item["run"])
        config = read(path / "config.json")
        runtime = read(path / "verl-runtime.json")
        status = read(path / "status.json")
        metrics = [json.loads(x) for x in (ROOT / path / "metrics.jsonl").read_text().splitlines()]
        expected = "opd" if name == "medical-opd" else name
        require(
            name + "/executed",
            item["passed"] and config["algorithm"] == expected and status["status"] == "completed",
        )
        workers = runtime["workers"]
        require(
            name + "/two_real_cpu_workers",
            len(workers) == 2
            and len({w["pid"] for w in workers}) == 2
            and all(w["device"] == "cpu" and w["world_size"] == 2 for w in workers)
            and runtime["driver_pid"] not in {w["pid"] for w in workers}
            and runtime["torch_cuda_build"] is None,
        )
        require(
            name + "/official_interfaces",
            runtime["verl"] == "0.7.1"
            and runtime["worker_class"] == "verl.single_controller.ray.RayWorkerGroup"
            and all(w["actor_class"] == "verl.workers.actor.dp_actor.DataParallelPPOActor" for w in workers),
        )
        require(
            name + "/saved_state",
            all(
                (ROOT / path / "checkpoint-final" / f).exists()
                for f in ["model/config.json", "controller.pt", "worker-rank-0.pt", "worker-rank-1.pt"]
            ),
        )
        require(
            name + "/logged_training",
            len(metrics) == config["steps"]
            and (ROOT / path / "trajectories.jsonl").exists()
            and any((ROOT / path).rglob("events.out.tfevents.*")),
        )
        hashes = runtime["final_parameters"]
        require(name + "/ranks_equal", hashes[0] == hashes[1])
        frozen = all(
            v == runtime["initial_parameters"][i][role]
            for i, roles in enumerate(hashes)
            for role, v in roles.items()
            if role != "actor"
        )
        require(name + "/frozen_models", frozen)
        changed = hashes[0]["actor"] != runtime["initial_parameters"][0]["actor"]
        require(
            name + "/parameter_update_or_documented_skip",
            changed or (name == "harness-rl" and all(m["update/skipped"] == 1 for m in metrics)),
        )
        if name in {"sar-opd", "idt-opd"}:
            require(name + "/two_teacher_phases", {m["distill/general_phase"] for m in metrics} == {0, 1})
        if name == "tempo":
            require(
                "tempo/replay_and_correction",
                any(m["tempo/replayed"] for m in metrics)
                and any(abs(m["tempo/prefix_is_weight"] - 1) > 1e-8 for m in metrics),
            )
        if name == "harness-rl":
            require(
                "harness/interface_and_trees",
                bool(list((ROOT / path).glob("interface-*.json")))
                and bool(list((ROOT / path).glob("trees-*.json"))),
            )
        evidence[name] = {
            "run": str(path),
            "workers": len(workers),
            "parameters_changed": changed,
            "max_grad_norm": max(m.get("update/grad_norm", 0) for m in metrics),
        }

    chapters = read("configs/chapters.json")
    gpu_configs = []
    for row in chapters:
        folder = ROOT / row["chapter"]
        cfg = yaml.safe_load((folder / "config.yaml").read_text())
        require(row["chapter"] + "/default_matches_registry", cfg["backend"] == row["backend"])
        if row["algorithm"] in SUPPORTED:
            for mode, device in [("verl", "cpu"), ("verl-gpu", "cuda")]:
                variant = yaml.safe_load((folder / (mode + ".yaml")).read_text())
                require(
                    row["chapter"] + "/" + mode, variant["backend"] == "verl" and variant["device"] == device
                )
                if device == "cuda":
                    gpu_configs.append(str((folder / (mode + ".yaml")).relative_to(ROOT)))
            require(row["chapter"] + "/guide", "verl 训练路径" in (folder / "README.md").read_text())
            if row["algorithm"] not in {"grpo", "gspo", "ppo"}:
                require(row["chapter"] + "/complex_defaults_to_verl", cfg["backend"] == "verl")
    fsdp = read("reports/verl-fsdp-capo.json")
    require(
        "actual_fsdp_capo",
        fsdp["passed"]
        and fsdp["sharding"] == "FULL_SHARD"
        and fsdp["parameters_changed"] > 0
        and fsdp["unselected_parameters_changed"] == 0
        and fsdp["rank_parameters_equal"]
        and fsdp["metrics"]["capo/routed_parameters"] > 0,
    )
    require(
        "fsdp_inference_and_official_checkpoint_restore",
        fsdp.get("fsdp_inference_reshard")
        and fsdp.get("official_checkpoint_restore_equal")
        and "REAL_CPU_FSDP_CAPO_PASSED" in (ROOT / "reports/verl-fsdp-capo.log").read_text(),
    )
    require("46_mechanism_tests", "46 passed" in (ROOT / "reports/verl-mechanism-tests.log").read_text())
    require(
        "cpu_medical_and_resume",
        resume["passed"]
        and resume["restored_weights_equal"]
        and resume["resumed_policy_version"] == 4
        and resume["medical_sft_to_verl"],
    )
    require("actual_textworld_with_verl", read("reports/verl-real-alfworld.json")["passed"])
    require("baseline_regression", all(x["passed"] for x in read("reports/cpu-audit-latest.json").values()))
    require(
        "baseline_checks", "All checks passed!" in (ROOT / "reports/final-baseline-check.log").read_text()
    )
    require(
        "official_gpu_interface_imports_on_cpu",
        '"passed": true' in (ROOT / "reports/verl-interface-verification.log").read_text(),
    )
    environments = {}
    for env in [".venv", ".venv-verl", ".venv-verl-gpu"]:
        check = subprocess.run(
            [str(ROOT / ".venv/bin/uv"), "pip", "check", "--python", str(ROOT / env / "bin/python")],
            capture_output=True,
            text=True,
        )
        require(env + "/pip_check", check.returncode == 0)
        code = "import json,importlib.metadata as m; print(json.dumps({x:m.version(x) for x in ['torch','trl','transformers','numpy']}))"
        versions = subprocess.check_output([str(ROOT / env / "bin/python"), "-c", code], text=True)
        environments[env] = json.loads(versions)
        require(
            env + "/torch_build",
            environments[env]["torch"] == ("2.9.0+cu129" if env.endswith("gpu") else "2.9.1+cpu"),
        )
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT / "references/verl"), "rev-parse", "HEAD"], text=True
    ).strip()
    require("official_verl_revision", revision == "bec9ef74768dd201881cd4e54cd0385e87caae27")
    locks = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ["uv.lock", "environments/verl-cpu/uv.lock", "environments/verl-gpu/uv.lock"]
    }
    for filename in [
        "README.md",
        "docs/VERL.md",
        "docs/ARCHITECTURE.md",
        "docs/RESEARCH.md",
        "docs/VALIDATION.md",
    ]:
        require(filename + "/verl_documented", "verl" in (ROOT / filename).read_text())
    report = {
        "passed": all(checks.values()),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "algorithms": evidence,
        "environments": environments,
        "locks_sha256": locks,
        "gpu_configs": gpu_configs,
        "gpu_execution": False,
        "limitations": [
            "No GPU or multi-node runtime validation",
            "No benchmark or throughput claim",
            "Harness random rollout has zero CAPO gradient; nonzero routing is verified separately with real FSDP",
            "CPU actor replicas use Gloo; the dedicated CPU FSDP experiment separately verifies FULL_SHARD",
        ],
    }
    (ROOT / "reports/verl-completion-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failed": [k for k, v in checks.items() if not v],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
