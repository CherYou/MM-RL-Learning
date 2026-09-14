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
MATH_FENCE = re.compile(
    r"^```math[ \t]*\n(.*?)^```[ \t]*$",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
SHELL_FENCE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANGLE_PLACEHOLDER = re.compile(r"<[^>\n]+>")
INLINE_CODE = re.compile(r"`[^`\n]*`")
GITHUB_INLINE_MATH = re.compile(r"\$`([^`\n]+)`\$")
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
MATH = re.compile(
    r"(?<!\\)\$\$(.*?)(?<!\\)\$\$|(?<![\\$])\$(?!\$)(.*?)(?<![\\$])\$(?!\$)",
    re.DOTALL,
)
TEX_ESCAPE = re.escape(chr(92))
UNSAFE_GITHUB_MATH = re.compile(
    TEX_ESCAPE + r"(?:operatorname|tfrac|newcommand|renewcommand|def|require)\b"
    + "|"
    + TEX_ESCAPE
    + r"(?:begin|end)\{cases\}"
    + "|"
    + TEX_ESCAPE
    + r"rm\b"
)


def is_beginner_material(relative):
    return (
        relative.endswith("/TUTORIAL.md")
        or relative == "TUTORIAL.md"
        or relative in {"docs/BEGINNER_GUIDE.md", "preliminary/FOUNDATIONS.md"}
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
    beginner_documents = 0
    beginner_formulas = 0
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
        plain_visible = INLINE_CODE.sub("", FENCED_CODE.sub("", text))
        if re.search(r"</?(?:answer|search|python|observation|action|value)>", plain_visible):
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
        beginner = is_beginner_material(relative)
        if beginner:
            beginner_documents += 1

        formulas = [match.group(1) for match in MATH_FENCE.finditer(text)]
        math_fence_starts = len(
            re.findall(r"^```math[ \t]*$", text, re.IGNORECASE | re.MULTILINE)
        )
        if math_fence_starts != len(formulas):
            findings.append(f"{relative}: unclosed or malformed math code fence")

        visible = FENCED_CODE.sub("", text)

        def collect_github_inline(match):
            formulas.append(match.group(1))
            return ""

        visible = GITHUB_INLINE_MATH.sub(collect_github_inline, visible)
        visible = INLINE_CODE.sub("", visible)
        legacy_formulas = []

        def collect(match):
            legacy_formulas.append(
                match.group(1) if match.group(1) is not None else match.group(2)
            )
            return ""

        remainder = MATH.sub(collect, visible)
        formulas.extend(legacy_formulas)
        if re.search(r"(?<!\\)\$", remainder):
            findings.append(f"{relative}: unmatched math delimiter")
        if beginner and legacy_formulas:
            findings.append(
                f"{relative}: beginner math must use $`...`$ inline syntax or a ```math fence"
            )
        if beginner:
            beginner_formulas += len(formulas)
        for formula in formulas:
            if error := brace_error(formula):
                findings.append(f"{relative}: {error} in {formula.strip()[:80]!r}")
            if unsafe := UNSAFE_GITHUB_MATH.search(formula):
                findings.append(
                    f"{relative}: GitHub-unsafe math syntax {unsafe.group()!r}"
                )
            if beginner and re.search(r"[<>]", formula):
                findings.append(
                    f"{relative}: use \\lt or \\gt instead of a raw angle bracket in math"
                )
            if re.search(r"y_\{(?:[^{}]*,)?<t\}", formula):
                findings.append(
                    f"{relative}: use y_{{1:t-1}} instead of an angle-bracket history subscript"
                )
    if findings:
        print("\n".join(findings))
        raise SystemExit(f"Found {len(findings)} Markdown math issue(s).")
    print(
        f"Markdown link and math checks passed for {checked} tracked documents; "
        f"{beginner_formulas} formulas in {beginner_documents} beginner materials "
        "use GitHub-safe delimiters."
    )


if __name__ == "__main__":
    main()
