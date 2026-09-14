#!/usr/bin/env python3
"""09-vision-grpo: local vision-grpo entry point. Run from any directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.cli import main

if __name__ == "__main__":
    main(["train", "09-vision-grpo/config.yaml", *sys.argv[1:]])
