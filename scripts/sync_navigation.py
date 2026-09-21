#!/usr/bin/env python3
"""Sync NAV:TOP/BOTTOM blocks in tutorials from configs/learning_paths.json.

Usage:
  python scripts/sync_navigation.py --write
  python scripts/sync_navigation.py --check
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_BEGIN = "<!-- NAV:TOP:BEGIN -->"
TOP_END = "<!-- NAV:TOP:END -->"
BOTTOM_BEGIN = "<!-- NAV:BOTTOM:BEGIN -->"
BOTTOM_END = "<!-- NAV:BOTTOM:END -->"

FAMILY_FILE = {
    "foundations": "docs/families/00-foundations.md",
    "policy-preference": "docs/families/01-policy-preference.md",
    "tools-multimodal": "docs/families/02-tools-multimodal.md",
    "distillation": "docs/families/03-distillation.md",
    "long-horizon": "docs/families/04-long-horizon.md",
    "control-offline": "docs/families/05-control-offline.md",
}


def load_registry():
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    paths = json.loads((ROOT / "configs/learning_paths.json").read_text(encoding="utf-8"))
    return chapters, paths


def rel(from_tutorial: str, target: str) -> str:
    start = posixpath.dirname(from_tutorial)
    return posixpath.relpath(target, start).replace("\\", "/")


def nav_line(tutorial: str, chapters_by_id: dict, book: list[str], index: int) -> str:
    row = chapters_by_id[book[index]]
    family_path = FAMILY_FILE[row["family"]]
    parts = []
    if index == 0:
        parts.append(f"[← 开始学习]({rel(tutorial, 'docs/START_HERE.md')})")
    else:
        prev = chapters_by_id[book[index - 1]]
        prev_t = prev["tutorial"]
        parts.append(
            f"[← 上一章：{prev['display_number']:02d} {prev['title']}]({rel(tutorial, prev_t)})"
        )
    parts.append(f"[全书目录]({rel(tutorial, 'docs/CHAPTERS.md')})")
    parts.append(f"[本篇目录]({rel(tutorial, family_path)})")
    if index + 1 < len(book):
        nxt = chapters_by_id[book[index + 1]]
        parts.append(
            f"[下一章：{nxt['display_number']:02d} {nxt['title']} →]({rel(tutorial, nxt['tutorial'])})"
        )
    else:
        parts.append(f"[课程回顾与独立实验 →]({rel(tutorial, 'docs/NEXT_STEPS.md')})")
    return " · ".join(parts)


def replace_block(text: str, begin: str, end: str, body: str, label: str) -> str:
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end),
        re.DOTALL,
    )
    block = f"{begin}\n{body}\n{end}"
    if pattern.search(text):
        return pattern.sub(lambda _m: block, text, count=1)
    return text  # caller inserts


def install_nav(path: Path, body_top: str, body_bottom: str, write: bool) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#"):
        errors.append(f"{path}: missing H1")
        return errors
    top_block = f"{TOP_BEGIN}\n{body_top}\n{TOP_END}"
    bottom_block = f"{BOTTOM_BEGIN}\n{body_bottom}\n{BOTTOM_END}"

    # Remove existing controlled blocks then reinsert after H1 and at EOF.
    text2 = re.sub(
        re.escape(TOP_BEGIN) + r".*?" + re.escape(TOP_END) + r"\n?",
        "",
        text,
        count=0,
        flags=re.DOTALL,
    )
    text2 = re.sub(
        re.escape(BOTTOM_BEGIN) + r".*?" + re.escape(BOTTOM_END) + r"\n?",
        "",
        text2,
        count=0,
        flags=re.DOTALL,
    )
    lines = text2.splitlines()
    if not lines:
        return [f"{path}: empty after strip"]
    body_start = 1
    while body_start < len(lines) and lines[body_start].strip() == "":
        body_start += 1
    body_lines = list(lines[body_start:])
    while body_lines and body_lines[-1].strip() == "":
        body_lines.pop()
    new_lines = [lines[0], "", top_block, ""] + body_lines + ["", bottom_block, ""]
    new_text = "\n".join(new_lines)
    if write:
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
    elif new_text != text:
        errors.append(f"{path.relative_to(ROOT)}: navigation out of date")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.write and not args.check:
        args.check = True

    chapters, paths = load_registry()
    by_id = {row["id"]: row for row in chapters}
    book = paths["book"]
    if len(book) != len(by_id):
        print("book length != registry size", file=sys.stderr)
        return 1

    errors: list[str] = []
    for i, chapter_id in enumerate(book):
        row = by_id[chapter_id]
        tutorial = row["tutorial"]
        path = ROOT / tutorial
        if not path.is_file():
            errors.append(f"missing tutorial {tutorial}")
            continue
        body = nav_line(tutorial, by_id, book, i)
        errors.extend(install_nav(path, body, body, write=args.write))

    # light validation
    for i, chapter_id in enumerate(book):
        row = by_id[chapter_id]
        for prereq in row.get("prerequisites", []):
            if prereq not in by_id:
                errors.append(f"{chapter_id}: unknown prerequisite {prereq}")
        for creq in row.get("concept_prerequisites", []):
            concepts = json.loads((ROOT / "configs/concepts.json").read_text(encoding="utf-8"))
            if creq not in concepts["concepts"]:
                errors.append(f"{chapter_id}: unknown concept {creq}")
            else:
                cpath = ROOT / concepts["concepts"][creq]["path"]
                if not cpath.is_file():
                    errors.append(f"concept path missing: {cpath}")

    if errors:
        print("\n".join(errors))
        return 1
    mode = "wrote" if args.write else "ok"
    print(f"{mode} navigation for {len(book)} chapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
