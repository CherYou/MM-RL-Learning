#!/usr/bin/env python3
"""Prepare real retrieval passages, medical evaluation and text-only game manifests."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, write_jsonl, question_key
from prepare_data import manifest


def retrieval(limit=256):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = "hotpotqa/hotpot_qa"
    revision = HfApi().dataset_info(repo).sha
    corpus = {}
    seen = set()
    for split, out in [("train", "train"), ("validation", "eval")]:
        ds = load_dataset(repo, "distractor", revision=revision, split=split, streaming=True)
        rows = []
        for r in itertools.islice(ds, limit):
            key = question_key(r["question"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "id": r["id"],
                    "prompt": r["question"],
                    "answer": r["answer"],
                    "answers": [r["answer"]],
                    "source": "HotpotQA distractor; local passage corpus",
                }
            )
            for title, sentences in zip(r["context"]["title"], r["context"]["sentences"], strict=True):
                text = " ".join(sentences)
                corpus[hashlib.sha256((title + text).encode()).hexdigest()] = {"title": title, "text": text}
        write_jsonl(ROOT / f"data/hotpot/{out}.jsonl", rows)
    # Corpus contains public passages, never question/answer annotations or gold support labels.
    write_jsonl(ROOT / "data/search/corpus.jsonl", list(corpus.values()))
    meta = {
        "repo": repo,
        "revision": revision,
        "config": "distractor",
        "queries_per_split": limit,
        "corpus": "Union of accompanying public distractor passages; no answer annotations",
        "license": "CC-BY-SA-4.0 (dataset card)",
        "scope": "Small local retrieval experiment; not full Wikipedia Search-R1 benchmark",
    }
    manifest(ROOT / "data/hotpot", meta)
    manifest(
        ROOT / "data/search",
        {**json.loads((ROOT / "data/search/manifest.json").read_text()), "corpus_source": meta},
    )
    print("Retrieval passages:", len(corpus), flush=True)


def medqa():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = "bigbio/med_qa"
    revision = HfApi().dataset_info(repo).sha
    ds = load_dataset(repo, "med_qa_zh_4options_source", revision=revision, split="test", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        options = r["options"]
        if isinstance(options, list):
            options = {
                str(x.get("key", chr(65 + j))): x.get("value", x.get("text", ""))
                for j, x in enumerate(options)
            }
        rows.append(
            {
                "id": f"medqa:{i}",
                "prompt": r["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items()),
                "answer": r["answer_idx"],
                "solution": r["answer"],
                "source": "MedQA Chinese four-option test",
            }
        )
    source_count = len(rows)
    unique = {}
    conflicts = set()
    for row in rows:
        key = question_key(row["prompt"])
        if key in unique and unique[key]["answer"] != row["answer"]:
            conflicts.add(key)
        unique.setdefault(key, row)
    rows = [row for key, row in unique.items() if key not in conflicts]
    write_jsonl(ROOT / "data/medqa/eval.jsonl", rows)
    manifest(
        ROOT / "data/medqa",
        {
            "repo": repo,
            "revision": revision,
            "split": "test",
            "license": "See source dataset card",
            "source_rows": source_count,
            "duplicate_rows": source_count - len(unique),
            "conflicting_questions": len(conflicts),
        },
    )
    print("MedQA evaluation:", len(rows), flush=True)


def alfworld(download=False):
    if download:
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

        dest = ROOT / "data/alfworld"
        dest.mkdir(parents=True, exist_ok=True)
        for release, name in [
            ("0.2.2", "json_2.1.1_json.zip"),
            ("0.2.2", "json_2.1.1_pddl.zip"),
            ("0.4.0", "json_2.1.2_tw-pddl.zip"),
        ]:
            archive = dest / name
            if not archive.exists():
                part = archive.with_suffix(".part")
                with requests.get(
                    f"https://github.com/alfworld/alfworld/releases/download/{release}/{name}",
                    stream=True,
                    timeout=120,
                ) as response:
                    response.raise_for_status()
                    with part.open("wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                part.rename(archive)
            with zipfile.ZipFile(archive) as z:
                if any(not (dest / n).resolve().is_relative_to(dest.resolve()) for n in z.namelist()):
                    raise ValueError("Unsafe archive path")
                z.extractall(dest)
        (dest / "logic").mkdir(exist_ok=True)
        shutil.copy(ALFRED_PDDL_PATH, dest / "logic/alfred.pddl")
        shutil.copy(ALFRED_TWL2_PATH, dest / "logic/alfred.twl2")
    from agentic_rl.alfworld_data import discover_games

    for split, out in [("train", "train"), ("valid_seen", "eval-seen"), ("valid_unseen", "eval-unseen")]:
        games = discover_games(ROOT / "data/alfworld", split)
        rows = [
            {
                "id": g.id,
                "prompt": g.id,
                "answer": "success",
                "split": g.split,
                "game_file": str(g.game_file.relative_to(ROOT)),
                "task_type": g.task_type,
            }
            for g in games
        ]
        write_jsonl(ROOT / f"data/alfworld-manifests/{out}.jsonl", rows)
        print("ALFWorld", split, len(rows), flush=True)
    manifest(
        ROOT / "data/alfworld-manifests",
        {
            "source": "https://github.com/alfworld/alfworld/releases",
            "versions": ["0.2.2 JSON/PDDL", "0.4.0 TW-PDDL"],
            "environment": "alfworld==0.4.2 / textworld==1.7.0",
            "scope": "all supported solvable text-only games",
            "license": "MIT (ALFWorld code); ALFRED data terms apply",
        },
    )
    skills = ROOT / "data/skills"
    skills.mkdir(exist_ok=True)
    original = ROOT / "references/upstream/09-AgentOPSD/skills/alfworld"
    for p in original.rglob("*.md"):
        shutil.copy2(p, skills / p.name)
    if not (skills / "general.md").exists():
        (skills / "general.md").write_text(
            "Find the target object. Open closed containers. Take the object before placing it. Follow admissible action names exactly.\n"
        )


def probes():
    rows = [
        {
            "prompt": "Calculate 2 + 3.",
            "response": json.dumps({"action": "Calculate", "args": "print(2 + 3)"}),
            "reward": 1.0,
            "provenance": "seed probe, manually specified and tool-verified; not an on-policy training trajectory",
        },
        {
            "prompt": "Calculate 2 + 3. Tool returned 5.",
            "response": json.dumps({"action": "Summary", "args": "5"}),
            "reward": 1.0,
            "provenance": "seed probe, manually specified; successful outcome",
        },
    ]
    write_jsonl(ROOT / "data/harness/probes.jsonl", rows)
    manifest(
        ROOT / "data/harness",
        {
            "source": "local verified seed probes",
            "purpose": "initialize CAPO unit masks; replace with successful collected trajectories for paper experiments",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["all", "retrieval", "medqa", "alfworld"], default="all")
    p.add_argument("--download-alfworld", action="store_true")
    a = p.parse_args()
    if a.only in {"all", "retrieval"}:
        retrieval()
    if a.only in {"all", "medqa"}:
        medqa()
    if a.only in {"all", "alfworld"}:
        alfworld(a.download_alfworld)
    probes()
