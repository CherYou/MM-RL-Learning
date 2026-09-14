#!/usr/bin/env python3
"""Verify root-CLI environment routing, medical SFT, and two-worker TEMPO resume on CPU."""

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "ACCELERATE_USE_CPU": "true"}
    evidence = {"device": "cpu", "passed": False, "commands": []}

    def command(name, args):
        log = ROOT / f"reports/verl-resume-{stamp}-{name}.log"
        cmd = [str(ROOT / ".venv/bin/arl"), *args]
        with log.open("w") as file:
            result = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT, timeout=420
            )
        evidence["commands"].append(
            {"command": cmd, "log": str(log.relative_to(ROOT)), "exit_code": result.returncode}
        )
        assert result.returncode == 0, log

    initial = f"runs/verl-resume-{stamp}-tempo"
    resumed = initial + "-continued"
    medical = f"runs/verl-resume-{stamp}-medical"
    command("tempo", ["train", "09-tempo/config.yaml", "--smoke", "--verl-workers", "2", "--output", initial])
    command(
        "continue",
        [
            "train",
            "09-tempo/config.yaml",
            "--smoke",
            "--verl-workers",
            "2",
            "--resume",
            initial + "/checkpoint-final",
            "--steps",
            "4",
            "--output",
            resumed,
        ],
    )
    before = json.loads((ROOT / initial / "verl-runtime.json").read_text())
    after = json.loads((ROOT / resumed / "verl-runtime.json").read_text())
    assert after["resume"]["restored_parameters"] == before["final_parameters"]
    assert after["resume"]["next_step"] == 3
    rows = [json.loads(line) for line in (ROOT / resumed / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 3
    assert rows[0]["verl/policy_version"] == 4
    command(
        "evaluate",
        [
            "eval",
            "09-tempo/config.yaml",
            "--smoke",
            "--checkpoint",
            resumed + "/checkpoint-final",
            "--limit",
            "1",
        ],
    )
    command(
        "medical",
        [
            "train",
            "02-opd/config.yaml",
            "--smoke",
            "--teacher-model",
            "medical_sft",
            "--verl-workers",
            "2",
            "--output",
            medical,
        ],
    )
    teacher = json.loads((ROOT / (medical + "-teacher-sft") / "status.json").read_text())
    student = json.loads((ROOT / medical / "status.json").read_text())
    config = json.loads((ROOT / medical / "config.json").read_text())
    assert teacher["status"] == student["status"] == "completed"
    assert config["teacher_model"].endswith("-teacher-sft/checkpoint-final/model")
    evidence.update(
        passed=True,
        initial_run=initial,
        resumed_run=resumed,
        medical_run=medical,
        restored_weights_equal=True,
        resumed_next_step=3,
        resumed_policy_version=4,
        evaluation_checkpoint_loaded=True,
        medical_sft_to_verl=True,
    )
    (ROOT / "reports/verl-resume.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
