#!/usr/bin/env python3
"""Make clearly labelled CPU fixtures from local data; never replace experiment data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, read_jsonl, write_jsonl
from prepare_data import fixtures, manifest

fixtures()
for split in ["train", "eval"]:
    rows = read_jsonl(f"data/fixtures/{split}.jsonl")
    write_jsonl(
        ROOT / f"data/fixtures/dpo-{split}.jsonl",
        [
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "chosen": r["solution"],
                "rejected": "<answer>wrong</answer>",
            }
            for r in rows
        ],
    )
    images = read_jsonl(f"data/geoqa/{split}.jsonl")
    write_jsonl(
        ROOT / f"data/fixtures/vision-{split}.jsonl",
        [
            {
                **r,
                "image": images[i]["image"],
                "prompt": f"Inspect image {split} {i}. Choose A or B.",
                "answer": "A",
                "source": "synthetic vision plumbing fixture; not GeoQA labels",
            }
            for i, r in enumerate(rows[:8])
        ],
    )
write_jsonl(
    ROOT / "data/fixtures/general-train.jsonl",
    [
        {
            "id": f"general:{i}",
            "prompt": f"What is {200 + i} + 1?",
            "answer": str(201 + i),
            "solution": str(201 + i),
        }
        for i in range(8)
    ],
)
manifest(
    ROOT / "data/fixtures",
    {
        "source": "generated CPU plumbing examples",
        "purpose": "check tensor/gradient/image/serialization paths only",
    },
)
