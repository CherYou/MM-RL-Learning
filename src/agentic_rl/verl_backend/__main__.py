"""Entry point used when the CLI switches to the separate verl environment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    from agentic_rl.cli import train
    from agentic_rl.data import report_path

    result = train(json.loads(Path(args.config_json).read_text()))
    print(f"Saved verl run: {report_path(result)}")


if __name__ == "__main__":
    main()
