#!/usr/bin/env python3
"""07-gspo: local gspo entry point. Run from any directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.cli import main

if __name__ == "__main__":
    main(["eval", "07-gspo/config.yaml", *sys.argv[1:]])
