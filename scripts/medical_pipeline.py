#!/usr/bin/env python3
"""Train a medical SFT teacher, then staged SAR-OPD or alternating IDT-OPD."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.cli import load_config, smoke_config, train


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["sar-opd", "idt-opd"], default="sar-opd")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=20)
    a = p.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    teacher = load_config("02-opd/sft.yaml", {"model": a.model, "steps": a.steps})
    if a.smoke:
        teacher = smoke_config(teacher)
    teacher["output"] = f"runs/medical-pipeline-{stamp}/teacher-sft"
    teacher_run = train(teacher)
    student = load_config("02-opd/config.yaml", {"model": a.model, "algorithm": a.variant, "steps": a.steps})
    if a.smoke:
        student = smoke_config(student)
    student["teacher_model"] = str(teacher_run / "checkpoint-final/model")
    student["output"] = f"runs/medical-pipeline-{stamp}/{a.variant}"
    train(student)
