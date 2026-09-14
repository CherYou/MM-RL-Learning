#!/usr/bin/env python3
"""Prepare tiny offline models or download a pinned pretrained model, without inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, report_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="tiny")
    a = p.parse_args()
    if a.model == "tiny":
        from agentic_rl.models import Policy, VisionPolicy

        Policy().save(ROOT / "models/tiny-text")
        VisionPolicy().save(ROOT / "models/tiny-vision")
        print("Saved offline tiny text and LLaVA checkpoints.")
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(a.model)
        destination = ROOT / "models" / a.model.replace("/", "--")
        snapshot_download(
            a.model,
            revision=info.sha,
            local_dir=destination,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.jinja",
                "README.md",
                "LICENSE*",
            ],
        )
        (destination / "download-manifest.json").write_text(
            json.dumps({"repo": a.model, "revision": info.sha, "inference_performed": False}, indent=2) + "\n"
        )
        print(report_path(destination))
