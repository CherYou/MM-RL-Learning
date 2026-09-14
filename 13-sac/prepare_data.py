#!/usr/bin/env python3
"""SAC chapter entry point; paths resolve from the repository root."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.embodied.runner import main  # noqa: E402

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
