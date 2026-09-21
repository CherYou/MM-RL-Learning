#!/usr/bin/env python3
"""Validate chapter registry, learning paths, concept anchors, and NAV blocks."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    paths = json.loads((ROOT / "configs/learning_paths.json").read_text(encoding="utf-8"))
    concepts = json.loads((ROOT / "configs/concepts.json").read_text(encoding="utf-8"))["concepts"]
    errors: list[str] = []

    ids = [row["id"] for row in chapters]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chapter ids")
    book = paths["book"]
    if set(book) != set(ids):
        errors.append("learning_paths.book ids != chapters.json ids")
    if book != ids:
        errors.append("chapters.json order must follow learning_paths.book")

    by_id = {row["id"]: row for row in chapters}
    for row in chapters:
        tpath = ROOT / row["tutorial"]
        if not tpath.is_file():
            errors.append(f"missing tutorial {row['tutorial']}")
        for key in ("prerequisites", "recommended_prerequisites"):
            for pid in row.get(key, []):
                if pid not in by_id:
                    errors.append(f"{row['id']}: unknown {key} {pid}")
        for cid in row.get("concept_prerequisites", []):
            if cid not in concepts:
                errors.append(f"{row['id']}: unknown concept {cid}")
            else:
                cfile = ROOT / concepts[cid]["path"]
                anchor = concepts[cid]["anchor"]
                if not cfile.is_file():
                    errors.append(f"concept file missing {cfile}")
                else:
                    text = cfile.read_text(encoding="utf-8")
                    if f'id="{anchor}"' not in text and f"name=\"{anchor}\"" not in text:
                        # also allow markdown heading anchors like ## foo {#anchor}
                        if f"{{#{anchor}}}" not in text and f"(#{anchor})" not in text:
                            if not re.search(
                                rf"^#+\s+.*\n(?:.*\n)*?<a id=\"{re.escape(anchor)}\"",
                                text,
                                re.M,
                            ) and f'<a id="{anchor}"' not in text:
                                errors.append(f"missing anchor #{anchor} in {concepts[cid]['path']}")

    for route_name, route in paths["routes"].items():
        seen: list[str] = []
        for rid in route["ids"]:
            if rid not in by_id:
                errors.append(f"route {route_name}: unknown id {rid}")
                continue
            for pre in by_id[rid].get("prerequisites", []):
                if pre not in route["ids"][: route["ids"].index(rid)] and pre not in seen:
                    # hard prereq must appear earlier in the same route or be loss-basics shared
                    if pre != "loss-basics" and pre not in seen:
                        errors.append(
                            f"route {route_name}: {rid} requires {pre} before it in the route"
                        )
            seen.append(rid)

    for row in chapters:
        tpath = ROOT / row["tutorial"]
        if not tpath.is_file():
            continue
        text = tpath.read_text(encoding="utf-8")
        if "<!-- NAV:TOP:BEGIN -->" not in text or "<!-- NAV:BOTTOM:BEGIN -->" not in text:
            errors.append(f"{row['tutorial']}: missing NAV blocks")
            continue
        top = re.search(r"<!-- NAV:TOP:BEGIN -->\n(.*?)\n<!-- NAV:TOP:END -->", text, re.S)
        bot = re.search(r"<!-- NAV:BOTTOM:BEGIN -->\n(.*?)\n<!-- NAV:BOTTOM:END -->", text, re.S)
        if not top or not bot:
            errors.append(f"{row['tutorial']}: malformed NAV markers")
        elif top.group(1) != bot.group(1):
            errors.append(f"{row['tutorial']}: TOP and BOTTOM nav differ")

    required_docs = [
        "docs/CHAPTERS.md",
        "docs/NEXT_STEPS.md",
        "docs/families/00-foundations.md",
        "docs/families/01-policy-preference.md",
        "docs/families/02-tools-multimodal.md",
        "docs/families/03-distillation.md",
        "docs/families/04-long-horizon.md",
        "docs/families/05-control-offline.md",
    ]
    for rel in required_docs:
        if not (ROOT / rel).is_file():
            errors.append(f"missing {rel}")

    if errors:
        print("\n".join(errors))
        return 1
    print(f"navigation ok: {len(chapters)} chapters, {len(paths['routes'])} routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
