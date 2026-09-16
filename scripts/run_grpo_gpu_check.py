#!/usr/bin/env python3
"""Run the repository's real GRPO validation on one selected GPU and record samples."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.data import report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-index",
        default=os.environ.get("ARL_GPU_INDEX", "0"),
        help="physical GPU index passed to CUDA_VISIBLE_DEVICES and nvidia-smi (default: 0)",
    )
    parser.add_argument("--output", default="runs/verl-grpo-gpu-" + time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    gpu_index = str(args.gpu_index)
    runs_root = (ROOT / "runs").resolve()
    requested_output = Path(args.output)
    run_dir = (
        requested_output if requested_output.is_absolute() else ROOT / requested_output
    ).resolve()
    try:
        relative_run = run_dir.relative_to(runs_root)
    except ValueError:
        parser.error("--output must resolve inside the repository's runs/ directory")
    if relative_run == Path("."):
        parser.error("--output must name a new subdirectory inside runs/")
    output = (Path("runs") / relative_run).as_posix()
    name = run_dir.name
    log = ROOT / "runs/.logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    mpl_cache = ROOT / "runs/.matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_index,
        "ACCELERATE_USE_CPU": "false",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "GLOO_SOCKET_IFNAME": "lo",
        "NCCL_SOCKET_IFNAME": "lo",
        "MPLCONFIGDIR": str(mpl_cache),
        # This functional check uses eager CUDA kernels. vLLM's logprob
        # helper otherwise invokes torch.compile despite enforce_eager=True,
        # which requires a usable host compiler/toolchain on this machine.
        "TORCHDYNAMO_DISABLE": "1",
    }
    command = [
        str(ROOT / ".venv-verl-gpu/bin/arl"),
        "train",
        "01-grpo/verify-gpu.yaml",
        "--output",
        output,
    ]
    samples = []
    started = time.monotonic()
    with log.open("w") as file:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
        while process.poll() is None:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    gpu_index,
                    "--query-gpu=index,uuid,memory.used,utilization.gpu",
                    "--format=csv,nounits",
                ],
                text=True,
            )
            row = next(csv.DictReader(output.splitlines(), skipinitialspace=True))
            samples.append(
                {
                    "seconds": time.monotonic() - started,
                    "physical_index": int(row["index"]),
                    "uuid": row["uuid"],
                    "memory_mib": int(row["memory.used [MiB]"]),
                    "utilization_percent": int(row["utilization.gpu [%]"]),
                }
            )
            time.sleep(2)
    report = {
        "command": [
            ".venv-verl-gpu/bin/arl",
            *command[1:-1],
            report_path(command[-1]),
        ],
        "exit_code": process.returncode,
        "log": log.relative_to(ROOT).as_posix(),
        "wall_seconds": time.monotonic() - started,
        "gpu_samples": samples,
        "sampled_peak_memory_mib": max((x["memory_mib"] for x in samples), default=0),
    }
    raw_target = run_dir / "gpu-monitor.json"
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    raw_target.write_text(json.dumps(report, indent=2) + "\n")
    public_report = {key: value for key, value in report.items() if key != "gpu_samples"}
    public_report["gpu_samples"] = [
        {key: value for key, value in sample.items() if key != "uuid"} for sample in samples
    ]
    public_target = ROOT / "reports" / (name + "-monitor.json")
    public_target.parent.mkdir(parents=True, exist_ok=True)
    public_target.write_text(json.dumps(public_report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in public_report.items() if key != "gpu_samples"}, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
