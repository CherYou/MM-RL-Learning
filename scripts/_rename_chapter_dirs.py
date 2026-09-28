"""Rename chapter folders to book display numbers and rewrite path references."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\nnnn\XiaomiMiMoProjects\rl-learning\MM-RL-Learning")

# old path (relative posix) -> new path
RENAMES = [
    ("01-foundations", "01-foundations"),
    ("02-ppo", "02-ppo"),
    ("03-grpo", "03-grpo"),
    ("04-dapo", "04-dapo"),
    ("05-gspo", "05-gspo"),
    ("06-dpo", "06-dpo"),
    ("07-arpo", "07-arpo"),
    ("08-aepo", "08-aepo"),
    ("09-search-r1", "09-search-r1"),
    ("10-retool", "10-retool"),
    ("11-alfworld", "11-alfworld"),
    ("12-vision-grpo", "12-vision-grpo"),
    ("13-general-opd", "13-general-opd"),
    ("14-opsd", "14-opsd"),
    ("15-medical-opd", "15-medical-opd"),
    ("16-agentopsd", "16-agentopsd"),
    ("17-tempo", "17-tempo"),
    ("18-harness-rl", "18-harness-rl"),
    ("19-td3", "19-td3"),
    ("20-sac", "20-sac"),
    ("21-her", "21-her"),
    ("22-iql", "22-iql"),
]


def git(*args: str) -> None:
    subprocess.run(["git", *args], cwd=ROOT, check=True)


# 1) Move nested first: general-opd out of 02-opd
gen_old = ROOT / "13-general-opd"
gen_new = ROOT / "13-general-opd"
if gen_old.exists() and not gen_new.exists():
    gen_new.parent.mkdir(parents=True, exist_ok=True)
    git("mv", "13-general-opd", "13-general-opd")
    print("moved 15-medical-opd/general-opd -> 13-general-opd")

for old, new in RENAMES:
    if old == "13-general-opd":
        continue
    src = ROOT / old
    dst = ROOT / new
    if src.exists() and not dst.exists():
        git("mv", old, new)
        print(f"mv {old} -> {new}")
    elif src.exists() and dst.exists():
        print(f"WARN both exist: {old} {new}")

# 2) Rewrite references in text files
# longer old names first to avoid partial clobber
ordered = sorted(RENAMES, key=lambda x: -len(x[0]))
patterns = []
for old, new in ordered:
    # path-like replacements
    patterns.append((old, new))

SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "runs", "reports", "__pycache__", "data", "references", "models"}

changed = 0
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    if path.suffix.lower() not in {
        ".md",
        ".py",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".txt",
        ".sh",
        ".ps1",
        ".cfg",
        ".ini",
    } and path.name not in {"Makefile", "NOTICE", "LICENSE"}:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    orig = text
    for old, new in ordered:
        # replace folder path tokens
        text = text.replace(f"{old}/", f"{new}/")
        text = text.replace(f"/{old}", f"/{new}")
        text = text.replace(f'"{old}"', f'"{new}"')
        text = text.replace(f"'{old}'", f"'{new}'")
        text = text.replace(f"({old})", f"({new})")
        # bare path references like 02-ppo/train.py already covered by first rule
        text = text.replace(f" {old}/", f" {new}/")
    if text != orig:
        path.write_text(text, encoding="utf-8")
        changed += 1
        print("updated", rel)

print("files changed", changed)

# 3) Special: learning_paths / registry already rewritten if they contained paths.
# Rebuild registry from builder after fixing AUTHORITATIVE_ORDER chapter fields.
reg = ROOT / "scripts/build_chapter_registry.py"
t = reg.read_text(encoding="utf-8")
for old, new in ordered:
    t = t.replace(f'"{old}/', f'"{new}/')
    t = t.replace(f'"{old}"', f'"{new}"')
reg.write_text(t, encoding="utf-8")
print("registry builder paths patched")
