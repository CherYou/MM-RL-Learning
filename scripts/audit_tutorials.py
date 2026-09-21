#!/usr/bin/env python3
"""Check tutorial coverage, navigation, image integrity and local link validity.

Validity gates only: structural completeness, reachable links, declared assets,
and math/code fence consistency. Style preferences (fixed chapter count, fixed
intro order, thanks placement, exactly one generated image, or banning nearby
source links) are intentionally not enforced.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CHAPTER_FIELDS = {"chapter", "algorithm", "dataset", "backend"}


def collect_markdown_documents(chapters: list[dict]) -> list[Path]:
    documents = [ROOT / "README.md"]
    for extra in (
        "docs/START_HERE.md",
        "docs/LEARNING_PATH.md",
        "docs/CHAPTERS.md",
        "docs/NEXT_STEPS.md",
        "docs/BEGINNER_GUIDE.md",
        "preliminary/FOUNDATIONS.md",
        "examples/math/README.md",
        "CONTRIBUTING.md",
    ):
        path = ROOT / extra
        if path.exists():
            documents.append(path)
    documents.extend(sorted((ROOT / "docs").glob("*.md")))
    documents.extend(sorted((ROOT / "docs" / "families").glob("*.md")))
    for row in chapters:
        folder = ROOT / row["chapter"]
        tutorial = ROOT / row.get("tutorial", folder / "TUTORIAL.md")
        readme = folder / "README.md"
        if tutorial.is_file():
            documents.append(tutorial)
        if readme.is_file():
            documents.append(readme)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in documents:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main() -> None:
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    assets = json.loads(
        (ROOT / "docs/assets/algorithms/manifest.json").read_text(encoding="utf-8")
    )["assets"]
    checks: dict[str, bool] = {}
    inventory: list[dict] = []

    checks["chapters_present"] = isinstance(chapters, list) and len(chapters) > 0
    checks["chapter_fields_complete"] = all(
        isinstance(row, dict) and REQUIRED_CHAPTER_FIELDS.issubset(row) for row in chapters
    )
    checks["chapter_paths_unique"] = len({row.get("chapter") for row in chapters}) == len(chapters)
    checks["retired_directories_absent"] = all(
        not (ROOT / name).exists() for name in ("00-loss-function", "10-ppo", "11-dpo")
    )
    checks["worklogs_removed"] = all(
        not (ROOT / path).exists()
        for path in ("docs/TUTORIAL_WORKLOG.md", "docs/VERL_IMPLEMENTATION_PLAN.md")
    )
    checks["assets_declared_unique"] = (
        isinstance(assets, list)
        and len(assets) > 0
        and len({row.get("sha256") for row in assets}) == len(assets)
        and len({row.get("file") for row in assets}) == len(assets)
    )

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documents = collect_markdown_documents(chapters)
    asset_files = {
        str((ROOT / "docs/assets/algorithms" / asset["file"]).resolve()) for asset in assets
    }

    for row in chapters:
        chapter = row["chapter"]
        folder = ROOT / chapter
        tutorial = folder / "TUTORIAL.md"
        chapter_readme = folder / "README.md"
        prefix = chapter + "/"

        checks[prefix + "tutorial_exists"] = tutorial.is_file()
        checks[prefix + "readme_exists"] = chapter_readme.is_file()
        if not tutorial.is_file() or not chapter_readme.is_file():
            continue

        text = tutorial.read_text(encoding="utf-8")
        readme_text = chapter_readme.read_text(encoding="utf-8")
        checks[prefix + "entry_link"] = bool(
            re.search(r"\[[^\]]+\]\(TUTORIAL\.md\)", readme_text)
        )
        checks[prefix + "entry_near_top"] = bool(
            re.search(
                r"\[[^\]]+\]\(TUTORIAL\.md\)",
                "\n".join(readme_text.splitlines()[:12]),
            )
        )
        checks[prefix + "root_tutorial_link"] = f"]({chapter}/TUTORIAL.md)" in root_readme

        math_fence_starts = len(
            re.findall(r"^```math[ \t]*$", text, re.IGNORECASE | re.MULTILINE)
        )
        math_fence_blocks = len(
            re.findall(
                r"^```math[ \t]*\n.*?^```[ \t]*$",
                text,
                re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        checks[prefix + "math_and_code"] = (
            math_fence_starts >= 1
            and math_fence_starts == math_fence_blocks
            and "$$" not in text
            and "```" in text
            and text.count("```") % 2 == 0
        )
        checks[prefix + "exercise"] = "练习" in text or "自测" in text

        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        image_paths = []
        image_ok = True
        for image in images:
            image_path = (folder / image).resolve()
            image_paths.append(str(image_path))
            if not image_path.is_file():
                image_ok = False
            elif asset_files and str(image_path) in asset_files:
                continue
            elif not image_path.is_file():
                image_ok = False
        # At least allow zero images; if images are present they must resolve.
        checks[prefix + "images_resolve"] = image_ok

        inventory.append(
            {
                "chapter": chapter,
                "tutorial_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "characters": len(text),
                "images": images,
            }
        )

    broken: list[list[str]] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            raw = link.strip()
            if raw.startswith("<") and ">" in raw:
                raw = raw[1 : raw.index(">")]
            else:
                raw = raw.split(maxsplit=1)[0]
            target = unquote(raw).split("#", 1)[0]
            if (
                target
                and not target.startswith(("http:", "https:", "mailto:", "data:"))
                and not (document.parent / target).exists()
            ):
                broken.append([document.relative_to(ROOT).as_posix(), target])
    checks["local_links_resolve"] = not broken
    checks["core_navigation_pages"] = all(
        (ROOT / path).is_file()
        for path in (
            "docs/START_HERE.md",
            "docs/LEARNING_PATH.md",
            "docs/CHAPTERS.md",
            "docs/NEXT_STEPS.md",
            "configs/learning_paths.json",
            "preliminary/TUTORIAL.md",
        )
    )

    for row in assets:
        path = ROOT / "docs/assets/algorithms" / row["file"]
        key = "image/" + str(row.get("id") or row["file"])
        if not path.is_file():
            checks[key] = False
            continue
        with Image.open(path) as im:
            checks[key] = list(im.size) == row["dimensions"] and (
                hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            )
            im.verify()

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "tutorials": inventory,
        "documents_checked": len(documents),
        "broken_links": broken,
        "review_scope": (
            "Structure/link/hash checks complement formula/code and visual review; "
            "they do not prove theoretical correctness"
        ),
        "policy": (
            "Validity gates only; fixed chapter counts, thanks placement, single-image "
            "quotas, and source-link bans are not enforced"
        ),
    }
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/tutorial-audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failed": [key for key, value in checks.items() if not value],
                "broken_links": broken,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
