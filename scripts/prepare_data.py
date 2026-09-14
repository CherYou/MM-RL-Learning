#!/usr/bin/env python3
"""Download fixed public sources; export a small, independent learning split.

Full originals stay in data/raw/cache. No random samples are described as benchmark
scores. --limit 0 consumes all source rows, except explicitly bounded smoke data.
"""

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentic_rl.data import ROOT, write_jsonl, question_key

SPECS = {
    "gsm8k": ("openai/gsm8k", "main", None),
    "deepmath": ("zwhe99/DeepMath-103K", None, None),
    "opsd": ("siyanzhao/Openthoughts_math_30k_opsd", None, "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"),
    "dapo": ("BytedTsinghua-SIA/DAPO-Math-17k", None, "65877096c24ffa7abc4e4fa5edb95cf3413a5674"),
    "aime25": ("yentinglin/aime_2025", None, "6f71d77b0b89b9dabe07ab466c51df33f514df7f"),
    "medical": ("FreedomIntelligence/medical-o1-reasoning-SFT", "zh", None),
    "search": ("PeterJinGo/nq_hotpotqa_train", None, None),
    "geoqa": ("hz2475/geoQA", None, None),
    "ceval": ("ceval/ceval-exam", "computer_network", None),
}


def normalize(key, i, row):
    prompt = row.get("question") or row.get("problem") or row.get("Question") or row.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = prompt[0]["content"]
    prefix = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
    prompt = str(prompt).removeprefix(prefix).removesuffix(suffix).strip()
    solution = row.get("solution") or row.get("Complex_CoT") or row.get("response") or row.get("answer") or ""
    answer = row.get("answer") or row.get("final_answer") or row.get("Response") or row.get("golden_answers")
    if not answer:
        answer = (row.get("reward_model") or {}).get("ground_truth", "")
    if isinstance(answer, dict):
        answer = answer.get("target", "")
    aliases = answer if isinstance(answer, list) else [str(answer or "")]
    answer = str(aliases[0]) if aliases else ""
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
        aliases = [answer]
    if not answer and solution:
        # OPSD only needs the privileged reference solution; verifier remains opt-in.
        import re

        matches = re.findall(r"\\boxed\{([^{}]+)\}", str(solution))
        answer = matches[-1] if matches else str(solution)
    if key == "ceval":
        prompt += "\n" + "\n".join(f"{v}. {row[v]}" for v in "ABCD")
    if not prompt or not answer:
        return None
    result = {
        "id": f"{key}:{row.get('id', i)}",
        "prompt": prompt,
        "answer": answer,
        "answers": [str(x) for x in aliases if x] or [answer],
        "solution": str(solution),
        "source": key,
    }
    if key == "geoqa":
        picture = row.get("image")
        if picture is None:
            images = row.get("images")
            picture = images[0] if images else None
        if picture is None:
            raise ValueError("GeoQA row has no image")
        image_path = ROOT / "data/geoqa/images" / f"{i:06d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        picture.convert("RGB").save(image_path)
        result["image"] = image_path.relative_to(ROOT).as_posix()
        result["original_split"] = row.get("original_split", "train")
    return result


def manifest(directory, meta):
    meta["files"] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            data = p.read_bytes()
            meta["files"][str(p.relative_to(directory))] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                **({"rows": len(data.splitlines())} if p.suffix == ".jsonl" else {}),
            }
    (directory / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def prepare(key, limit):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo, config, revision = SPECS[key]
    info = HfApi().dataset_info(repo, revision=revision)
    revision = info.sha
    directory = ROOT / "data" / key
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo": repo,
        "revision": revision,
        "config": config,
        "seed": 42,
        "requested_limit": limit,
        "license": (info.card_data or {}).get("license", "See source dataset card"),
        "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
    }
    rows, conflicts = {}, set()
    splits = ["train", "test"] if key in {"gsm8k", "search"} else ["val"] if key == "ceval" else ["train"]
    collected = {}
    for split in splits:
        if key == "search":
            # Source train/test extra_info structs have incompatible schemas.
            # Project the stable columns before Arrow tries to cast those structs.
            files = [f.rfilename for f in info.siblings if f.rfilename.endswith(f"{split}.parquet")]
            if len(files) != 1:
                raise ValueError(f"Expected one {split}.parquet in {repo}: {files}")
            ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/{repo}@{revision}/{files[0]}",
                columns=["question", "golden_answers"],
                split="train",
                streaming=True,
            )
        else:
            ds = load_dataset(
                repo,
                name=config,
                revision=revision,
                split=split,
                streaming=True,
                cache_dir=str(ROOT / "data/raw/cache"),
            )
        source = ds if limit == 0 else itertools.islice(ds, limit)
        count = 0
        split_rows = []
        for i, row in enumerate(source):
            count += 1
            if key == "geoqa":
                import ast

                choices = (
                    ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
                )
                label = int(row["label"])
                row = {
                    **row,
                    "question": row["subject"]
                    + "\n"
                    + "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(choices)),
                    "solution": row["answer"],
                    "answer": chr(65 + label),
                }
            normalized = normalize(key, i, row)
            if normalized is None:
                continue
            k = question_key(normalized["prompt"])
            if k in rows:
                if rows[k]["answer"] != normalized["answer"]:
                    conflicts.add(k)
                continue
            rows[k] = normalized
            split_rows.append(normalized)
        collected[split] = split_rows
        meta[f"source_{split}_rows_read"] = count
    for split in collected:
        collected[split] = [x for x in collected[split] if question_key(x["prompt"]) not in conflicts]
    if key == "aime25":
        exported = {"eval": collected["train"]}
    elif key in {"gsm8k", "search"}:
        exported = {"train": collected["train"], "eval": collected["test"]}
    elif key == "geoqa":
        exported = {
            "train": [x for x in rows.values() if x["original_split"] == "train"],
            "eval": [x for x in rows.values() if x["original_split"] == "test"],
        }
    else:
        all_rows = next(iter(collected.values()))
        random.Random(42).shuffle(all_rows)
        n = max(1, min(64, len(all_rows) // 5))
        exported = {"train": all_rows[n:], "eval": all_rows[:n]}
    if any(not x for x in exported.values()):
        raise ValueError(f"{key}: an exported split is empty; increase --limit or check schema")
    for split, selected in exported.items():
        write_jsonl(directory / f"{split}.jsonl", selected)
    if key == "gsm8k":
        # Transparent synthetic negatives derived ONLY within each corresponding split.
        for split, selected in exported.items():
            pairs = []
            for row in selected:
                try:
                    wrong = str(float(row["answer"].replace(",", "")) + 1)
                except ValueError:
                    wrong = "I do not know."
                pairs.append(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "chosen": row["solution"],
                        "rejected": f"The answer is {wrong}.",
                        "source": "GSM8K solution vs synthetic wrong answer; learning fixture",
                    }
                )
            write_jsonl(ROOT / "data/dpo" / f"{split}.jsonl", pairs)
        manifest(ROOT / "data/dpo", {"derived_from": meta, "negative_generation": "numeric answer + 1"})
    meta["conflicting_questions_removed"] = len(conflicts)
    meta["scope"] = (
        "full requested source splits"
        if not limit
        else "bounded source prefix; learning subset, not full benchmark"
    )
    manifest(directory, meta)
    print(
        json.dumps({"dataset": key, "revision": revision, "rows": {k: len(v) for k, v in exported.items()}}),
        flush=True,
    )


def fixtures():
    """Offline arithmetic/debug fixtures, explicitly separate from public experiments."""
    for split, offset in [("train", 0), ("eval", 100)]:
        rows = [
            {
                "id": f"fixture:{offset + i}",
                "prompt": f"What is {offset + i} + {i + 1}?",
                "answer": str(offset + 2 * i + 1),
                "solution": f"<answer>{offset + 2 * i + 1}</answer>",
            }
            for i in range(32)
        ]
        write_jsonl(ROOT / "data/fixtures" / f"{split}.jsonl", rows)
    manifest(
        ROOT / "data/fixtures",
        {"source": "locally generated arithmetic", "purpose": "CPU plumbing checks only"},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=[*SPECS, "all", "fixtures"], default="all")
    p.add_argument("--limit", type=int, default=1024, help="Rows read per source split; 0 means all")
    args = p.parse_args()
    fixtures()
    failures = {}
    for key in SPECS if args.dataset == "all" else [] if args.dataset == "fixtures" else [args.dataset]:
        try:
            prepare(key, args.limit)
        except Exception as error:
            failures[key] = {"error_type": type(error).__name__}
            print(f"FAILED {key}: {failures[key]}", flush=True)
    if failures:
        (ROOT / "reports/data-download-errors.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(1)
