#!/usr/bin/env python3
"""Check local links and common Markdown math rendering failures."""

from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SKIPPED_PREFIXES = ("references/upstream/",)
FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
SHELL_FENCE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANGLE_PLACEHOLDER = re.compile(r"<[^>\n]+>")
INLINE_CODE = re.compile(r"`[^`\n]*`")
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
MATH = re.compile(
    r"(?<!\\)\$\$(.*?)(?<!\\)\$\$|(?<![\\$])\$(?!\$)(.*?)(?<![\\$])\$(?!\$)",
    re.DOTALL,
)


def is_escaped(text, index):
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def brace_error(formula):
    depth = 0
    for index, char in enumerate(formula):
        if char == "{" and not is_escaped(formula, index):
            depth += 1
        elif char == "}" and not is_escaped(formula, index):
            depth -= 1
            if depth < 0:
                return "extra closing brace"
    return "missing closing brace" if depth else None


def main():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    names = [name for name in tracked if name.lower().endswith(".md")]
    entries = set(tracked)
    for name in tracked:
        parent = Path(name).parent
        while parent != Path("."):
            entries.add(parent.as_posix())
            parent = parent.parent
    findings = []
    checked = 0
    for relative in names:
        if relative.startswith(SKIPPED_PREFIXES):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        checked += 1
        for block in SHELL_FENCE.finditer(text):
            for placeholder in ANGLE_PLACEHOLDER.finditer(block.group(1)):
                offset = block.start(1) + placeholder.start()
                line_number = text.count("\n", 0, offset) + 1
                findings.append(
                    f"{relative}:{line_number}: shell placeholder "
                    f"{placeholder.group()!r} is parsed as redirection"
                )
        if any(line.strip() == "$$" for line in text.splitlines()):
            findings.append(f"{relative}: standalone $$ delimiter; keep each display formula self-contained")
        visible = INLINE_CODE.sub("", FENCED_CODE.sub("", text))
        if re.search(r"</?(?:answer|search|python|observation|action|value)>", visible):
            findings.append(f"{relative}: wrap literal model tags in backticks")
        for raw_link in LOCAL_LINK.findall(text):
            raw_link = raw_link.strip()
            if raw_link.startswith("<") and ">" in raw_link:
                raw_link = raw_link[1 : raw_link.index(">")]
            else:
                raw_link = raw_link.split(maxsplit=1)[0]
            link = unquote(raw_link).split("#", 1)[0]
            if not link or link.startswith(("http:", "https:", "mailto:", "data:")):
                continue
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(relative), link.replace("\\", "/"))
            )
            if target not in entries:
                findings.append(f"{relative}: broken or incorrectly cased local link {raw_link!r}")
        formulas = []

        def collect(match):
            formulas.append(match.group(1) if match.group(1) is not None else match.group(2))
            return ""

        remainder = MATH.sub(collect, visible)
        if re.search(r"(?<!\\)\$", remainder):
            findings.append(f"{relative}: unmatched math delimiter")
        for formula in formulas:
            if error := brace_error(formula):
                findings.append(f"{relative}: {error} in {formula.strip()[:80]!r}")
            if re.search(r"y_\{(?:[^{}]*,)?<t\}", formula):
                findings.append(
                    f"{relative}: use y_{{1:t-1}} instead of an angle-bracket history subscript"
                )
    if findings:
        print("\n".join(findings))
        raise SystemExit(f"Found {len(findings)} Markdown math issue(s).")
    print(f"Markdown link and math checks passed for {checked} tracked documents.")


if __name__ == "__main__":
    main()
