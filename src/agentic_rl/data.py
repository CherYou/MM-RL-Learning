"""Shared JSONL contracts and provenance-aware local dataset loading."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]


def report_path(path):
    """Return a machine-independent path for reports and user-facing output."""
    raw = os.fspath(path)
    native = Path(raw)
    if native.is_absolute():
        try:
            return native.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return f"<external>/{native.name}"
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if posix.is_absolute():
        return f"<external>/{posix.name}"
    if windows.is_absolute():
        return f"<external>/{windows.name}"
    if str(raw).lower().startswith("file://"):
        return f"<external>/{PurePosixPath(str(raw).replace(chr(92), '/')).name}"
    return native.as_posix()


def read_jsonl(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    # U+2028/U+0085 can occur inside valid JSON strings; only LF delimits JSONL.
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def question_key(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def validate_dataset(path, require_answer=True):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if not row.get("prompt") or (require_answer and not row.get("answer")):
            raise ValueError(f"Missing prompt or answer: {row.get('id')}")
        key = question_key(row["prompt"])
        if key in seen:
            raise ValueError(f"Duplicate prompt: {row.get('id')}")
        seen.add(key)
        if row.get("image") and not (ROOT / row["image"]).is_file():
            raise FileNotFoundError(row["image"])
    return rows, seen
