#!/usr/bin/env python3
"""Eight general-domain C-Eval subsets; use labelled dev/val and split by question."""

import random
import sys
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, write_jsonl, question_key
from prepare_data import manifest

CONFIGS = [
    "computer_network",
    "college_programming",
    "advanced_mathematics",
    "discrete_mathematics",
    "college_physics",
    "logic",
    "chinese_language_and_literature",
    "college_economics",
]


def main():
    repo = "ceval/ceval-exam"
    revision = HfApi().dataset_info(repo).sha
    train = []
    evaluation = []
    seen = set()
    for name in CONFIGS:
        rows = []
        for split in ["dev", "val"]:
            ds = load_dataset(repo, name, revision=revision, split=split, streaming=True)
            for row in ds:
                prompt = row["question"] + "\n" + "\n".join(f"{k}. {row[k]}" for k in "ABCD")
                key = question_key(prompt)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "id": f"ceval:{name}:{split}:{row['id']}",
                        "prompt": prompt,
                        "answer": row["answer"],
                        "solution": row.get("explanation", "") or row["answer"],
                        "source": name,
                        "original_split": split,
                    }
                )
        random.Random(42).shuffle(rows)
        n = max(1, len(rows) // 5)
        evaluation.extend(rows[:n])
        train.extend(rows[n:])
    write_jsonl(ROOT / "data/ceval/train.jsonl", train)
    write_jsonl(ROOT / "data/ceval/eval.jsonl", evaluation)
    manifest(
        ROOT / "data/ceval",
        {
            "repo": repo,
            "revision": revision,
            "configs": CONFIGS,
            "source_splits": ["dev", "val"],
            "seed": 42,
            "split": "80/20 within each subject after global prompt deduplication",
            "license": "CC-BY-NC-SA-4.0 (source card)",
            "scope": "labelled learning subset, not official C-Eval test leaderboard",
        },
    )
    print("C-Eval", len(train), len(evaluation), flush=True)


if __name__ == "__main__":
    main()
