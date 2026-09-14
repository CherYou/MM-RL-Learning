#!/usr/bin/env python3
"""02-opd/general-opd: local opd entry point. Run from any directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.cli import main

if __name__ == "__main__":
    main(["eval", "02-opd/general-opd/config.yaml", *sys.argv[1:]])
