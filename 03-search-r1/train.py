#!/usr/bin/env python3
"""03-search-r1: local search-r1 entry point. Run from any directory."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl.cli import main

if __name__ == "__main__":
    main(["train", "03-search-r1/config.yaml", *sys.argv[1:]])
