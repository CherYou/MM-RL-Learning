#!/usr/bin/env python3
"""Check tutorial coverage, navigation, image integrity and source-link consolidation."""

import hashlib
import json
from pathlib import Path
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    assets = json.loads(
        (ROOT / "docs/assets/algorithms/manifest.json").read_text(encoding="utf-8")
    )["assets"]
    checks, inventory = {}, []
    checks["twenty_chapters"] = len(chapters) == 20
    checks["intro_ppo_dpo_order"] = [row["chapter"] for row in chapters[:3]] == [
        "preliminary",
        "001-ppo",
        "002-dpo",
    ]
    checks["retired_directories_absent"] = all(
        not (ROOT / name).exists() for name in ("00-loss-function", "10-ppo", "11-dpo")
    )
    checks["twenty_generated_assets"] = len(assets) == 20 and len({row["sha256"] for row in assets}) == 20
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checks["thanks_at_start"] = root_readme.index("## 参考与感谢") < root_readme.index("## 这套课程包含什么")
    checks["worklogs_removed"] = all(
        not (ROOT / path).exists()
        for path in ("docs/TUTORIAL_WORKLOG.md", "docs/VERL_IMPLEMENTATION_PLAN.md")
    )
    documents = [ROOT / "README.md", *ROOT.glob("docs/*.md"), ROOT / "preliminary/FOUNDATIONS.md"]
    asset_files = {str((ROOT / "docs/assets/algorithms" / asset["file"]).resolve()) for asset in assets}
    for row in chapters:
        folder = ROOT / row["chapter"]
        tutorial = folder / "TUTORIAL.md"
        text = tutorial.read_text(encoding="utf-8")
        documents.extend([tutorial, folder / "README.md"])
        prefix = row["chapter"] + "/"
        chapter_readme = (folder / "README.md").read_text(encoding="utf-8")
        checks[prefix + "entry_link"] = bool(
            re.search(r"\[[^\]]+\]\(TUTORIAL\.md\)", chapter_readme)
        )
        checks[prefix + "entry_near_top"] = bool(
            re.search(
                r"\[[^\]]+\]\(TUTORIAL\.md\)",
                "\n".join(chapter_readme.splitlines()[:12]),
            )
        )
        checks[prefix + "root_tutorial_link"] = (
            f"]({row['chapter']}/TUTORIAL.md)" in root_readme
        )
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
        checks[prefix + "exercise"] = "练习" in text
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        checks[prefix + "generated_image"] = (
            len(images) == 1 and str((folder / images[0]).resolve()) in asset_files
        )
        checks[prefix + "repo_links_consolidated"] = all(
            "https://github.com/" not in doc.read_text(encoding="utf-8")
            and "references/upstream" not in doc.read_text(encoding="utf-8")
            for doc in folder.glob("*.md")
        )
        inventory.append(
            {
                "chapter": row["chapter"],
                "tutorial_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "characters": len(text),
                "image": images,
            }
        )
    broken = []
    for document in documents:
        for link in re.findall(
            r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")
        ):
            link = link.strip("<>").split("#", 1)[0]
            if (
                link
                and not link.startswith(("http:", "https:", "mailto:"))
                and not (document.parent / link).exists()
            ):
                broken.append([document.relative_to(ROOT).as_posix(), link])
    checks["local_links_resolve"] = not broken
    for row in assets:
        path = ROOT / "docs/assets/algorithms" / row["file"]
        with Image.open(path) as im:
            checks["image/" + row["id"]] = (
                list(im.size) == row["dimensions"]
                and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            )
            im.verify()
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "tutorials": inventory,
        "documents_checked": len(documents),
        "broken_links": broken,
        "review_scope": "Structure/link/hash checks complement formula/code and visual review; they do not prove theoretical correctness",
    }
    (ROOT / "reports/tutorial-audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(checks),
                "failed": [key for key, value in checks.items() if not value],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
