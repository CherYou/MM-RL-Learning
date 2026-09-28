#!/usr/bin/env python3
"""CPU demos for DAPO-Tool / ARPO / AEPO teaching cores.

Usage:
  python examples/math/agentic_rl_demo.py
  python examples/math/agentic_rl_demo.py --algorithm arpo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rl.aepo import demo as aepo_demo  # noqa: E402
from agentic_rl.arpo import demo as arpo_demo  # noqa: E402

DEMOS = {
    "arpo": arpo_demo,
    "aepo": aepo_demo,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=sorted(DEMOS) + ["all"],
        default="all",
    )
    args = parser.parse_args()
    keys = sorted(DEMOS) if args.algorithm == "all" else [args.algorithm]
    out = {k: DEMOS[k]() for k in keys}
    out["status"] = "passed"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
