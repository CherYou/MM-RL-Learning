#!/usr/bin/env python3
"""Inventory authored code and flag exact token passages against the pinned reference.

This is a reproducible review aid, not a universal originality/copyright detector.
It cannot identify copied material from sources outside the comparison corpus.
"""

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/upstream"
ACKNOWLEDGED = {
    "src/agentic_rl/agentopsd.py",
    "src/agentic_rl/alfworld_data.py",
    "src/agentic_rl/alfworld_env.py",
}


def tokens(path):
    ignored = {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    return [
        (item.string, item.start[0])
        for item in tokenize.generate_tokens(
            io.StringIO(path.read_text(encoding="utf-8")).readline
        )
        if item.type not in ignored
    ]


def own_files():
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    directories = [ROOT / name for name in ("src", "scripts", "tests", "environments")]
    directories += [ROOT / row["chapter"] for row in chapters]
    return sorted(
        {
            path
            for folder in directories
            for path in folder.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"} and ".venv" not in path.parts
        }
    )


def main():
    reference = {path: tokens(path) for path in REFERENCE.rglob("*.py")}
    lookup = defaultdict(set)
    window = 50
    for path, sequence in reference.items():
        values = [value for value, line in sequence]
        for index in range(len(values) - window + 1):
            lookup[tuple(values[index : index + window])].add(path)
    inventory, flagged = [], []
    for path in own_files():
        relative = path.relative_to(ROOT).as_posix()
        inventory.append(
            {
                "file": relative,
                "sha256": hashlib.sha256(
                    path.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest(),
                "language": path.suffix,
                "historical_port": relative in ACKNOWLEDGED,
            }
        )
        if path.suffix != ".py":
            continue
        sequence = tokens(path)
        values = [value for value, line in sequence]
        candidates = set()
        for index in range(len(values) - window + 1):
            candidates.update(lookup.get(tuple(values[index : index + window]), ()))
        for candidate in sorted(candidates):
            original = reference[candidate]
            matcher = SequenceMatcher(None, values, [value for value, line in original], autojunk=False)
            for match in matcher.get_matching_blocks():
                if match.size >= window:
                    flagged.append(
                        {
                            "file": relative,
                            "line": sequence[match.a][1],
                            "reference": candidate.relative_to(ROOT).as_posix(),
                            "reference_line": original[match.b][1],
                            "tokens": match.size,
                            "acknowledged_port": relative in ACKNOWLEDGED,
                        }
                    )
    # Paragraph comparison deliberately skips equations and short shared terminology.
    originals = {
        path: path.read_text(encoding="utf-8", errors="replace")
        for path in REFERENCE.rglob("*.md")
    }
    text_matches = []
    guides = sorted(ROOT.glob("*/TUTORIAL.md")) + sorted(ROOT.glob("*/*/TUTORIAL.md"))
    guides += [ROOT / "preliminary/FOUNDATIONS.md"]
    for guide in guides:
        for paragraph in guide.read_text(encoding="utf-8").split("\n\n"):
            if sum("\u4e00" <= char <= "\u9fff" for char in paragraph) < 50:
                continue
            for source, content in originals.items():
                if paragraph in content:
                    text_matches.append(
                        {
                            "file": guide.relative_to(ROOT).as_posix(),
                            "reference": source.relative_to(ROOT).as_posix(),
                            "paragraph": paragraph,
                        }
                    )
    report = {
        "comparison": "Pinned references/upstream only",
        "minimum_exact_python_tokens": window,
        "inventory": inventory,
        "flagged_passages": sorted(flagged, key=lambda row: -row["tokens"]),
        "tutorials_scanned": len(guides),
        "identical_long_chinese_paragraphs": text_matches,
        "limits": [
            "Token matches need human review; formulas, API names and boilerplate can coincide",
            "No claim of global zero plagiarism or legal clearance",
            "Shell files inventoried by hash and reviewed separately; Python token scan only",
            "Third-party dependency/reference snapshots remain attributed external material",
        ],
    }
    target = ROOT / "reports/code-provenance-audit.json"
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(inventory),
                "python_passages": len(flagged),
                "unacknowledged_files": sorted(
                    {row["file"] for row in flagged if not row["acknowledged_port"]}
                ),
                "identical_tutorial_paragraphs": len(text_matches),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
