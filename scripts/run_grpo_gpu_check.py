#!/usr/bin/env python3
"""Run the repository's real GRPO validation on physical GPU 1 and record GPU samples."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/verl-grpo-gpu1-" + time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    name = Path(args.output).name
    log = ROOT / "reports" / (name + ".log")
    mpl_cache = ROOT / "runs/.matplotlib"
    mpl_cache.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "1",
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
        "01-grpo/verify-gpu1.yaml",
        "--output",
        args.output,
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
                    "1",
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
        "command": command,
        "exit_code": process.returncode,
        "log": str(log.relative_to(ROOT)),
        "wall_seconds": time.monotonic() - started,
        "gpu_samples": samples,
        "sampled_peak_memory_mib": max((x["memory_mib"] for x in samples), default=0),
    }
    target = ROOT / "reports" / (name + "-monitor.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "gpu_samples"}, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
