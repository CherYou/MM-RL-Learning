#!/usr/bin/env python3
"""Reject machine paths and hardware identifiers in tracked project files."""

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).relative_to(ROOT).as_posix()
SKIPPED_PREFIXES = ("references/upstream/",)
PORTABLE_PLACEHOLDERS = re.compile(
    r"\$\{(?:PROJECT_ROOT|HOME|TMPDIR|PYTHON_STDLIB|VLLM_SOURCE|PYTORCH_SOURCE)\}"
)
CHECKS = {
    "machine-specific Unix path": re.compile(
        r"(?<![.:/A-Za-z0-9_])/(?:"
        r"(?:data|home|root|mnt|workspace|tmp|Users|opt|srv|scratch|"
        r"gpfs|lustre|nfs|nas|ceph|cluster|localdisk|pytorch|shared|fsx|"
        r"raid|Volumes|project)/|"
        r"usr/(?:lib|local)/|var/(?:lib|log|tmp)/)"
    ),
    "machine-specific file URI": re.compile(r"file:///", re.IGNORECASE),
    "JSON-escaped Windows absolute path": re.compile(
        r"\b[A-Za-z]:(?:/|\\\\|\\(?![abfnrtv]))", re.IGNORECASE
    ),
    "private network address": re.compile(
        r"(?:(?<![A-Za-z0-9])_?(?:ip|ip[_-]?address|node[_-]?ip[_-]?address|"
        r"address|host|hostname|server|ray[_-]address|master[_-]?addr)"
        r"(?![A-Za-z0-9])[\"']?\s*"
        r"(?:[:=]\s*|\s+)[\"']?|(?:ssh|scp|sftp|rsync)\b[^\r\n]{0,80}|"
        r"(?:https?|ray)://)"
        r"(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        re.IGNORECASE,
    ),
    "GPU hardware UUID": re.compile(
        r"(?:\bGPU-|[\"']?(?:device_uuid|uuid)[\"']?\s*[:=]\s*[\"']?)"
        r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    "SSH host name": re.compile(
        r"^\s*Host(?:Name)?\s+(?!\*\s*$|localhost\b|\$\{)[^\s#]+", re.IGNORECASE
    ),
    "recorded host name": re.compile(
        r"[\"']?(?:host|hostname)[\"']?\s*[:=]\s*[\"']?"
        r"(?!\$\{HOSTNAME\}|localhost\b|127\.0\.0\.1\b|0\.0\.0\.0\b)"
        r"[A-Za-z0-9][A-Za-z0-9._-]+",
        re.IGNORECASE,
    ),
    "remote host in shell command": re.compile(
        r"^\s*(?:ssh|scp|sftp|rsync)\b[^\r\n]{0,120}\b"
        r"(?:[A-Za-z0-9._-]+@)?(?!localhost\b|github\.com\b)"
        r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b",
        re.IGNORECASE,
    ),
}
PLAIN_WINDOWS_ABSOLUTE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")
PLAIN_UNC_ABSOLUTE_PATH = re.compile(
    r"(?<!\\)\\{2,}[A-Za-z0-9._-]+\\{1,}[A-Za-z0-9$_.-]+", re.IGNORECASE
)
JSON_UNC_ABSOLUTE_PATH = re.compile(
    r'"(?:path|file|directory|root|checkpoint|model|dataset|output|log|saved|original|'
    r'image|cwd|workdir|working[_-]dir)"'
    r'\s*:\s*"\\{4,}[A-Za-z0-9._-]+\\{2,}[A-Za-z0-9$_.-]+',
    re.IGNORECASE,
)
REPORT_ABSOLUTE_PATH = re.compile(
    r"(?<![.:/A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+"
)


def violation_labels(relative, line):
    """Return portability rule labels for one tracked text line."""
    checked_line = PORTABLE_PLACEHOLDERS.sub(".", line)
    labels = [label for label, pattern in CHECKS.items() if pattern.search(checked_line)]
    if (
        Path(relative).suffix.lower() not in {".json", ".jsonl"}
        and PLAIN_WINDOWS_ABSOLUTE_PATH.search(checked_line)
        and not any("Windows absolute path" in label for label in labels)
    ):
        labels.append("Windows absolute path")
    if Path(relative).suffix.lower() in {".json", ".jsonl"}:
        if JSON_UNC_ABSOLUTE_PATH.search(checked_line):
            labels.append("UNC absolute path")
    elif PLAIN_UNC_ABSOLUTE_PATH.search(checked_line):
        labels.append("UNC absolute path")
    if relative.startswith("reports/") and REPORT_ABSOLUTE_PATH.search(checked_line):
        labels.append("absolute path in public report")
    return labels


def verify_rules():
    """Exercise high-risk matches and known false-positive boundaries."""
    gpu_id = "12345678-1234-1234-1234-123456789abc"
    positives = [
        ("docs/example.md", "/gpfs/team/project/model", "machine-specific Unix path"),
        ("docs/example.md", r"C:\temp\model", "Windows absolute path"),
        ("docs/example.md", r"\\server\share\model", "UNC absolute path"),
        ("data/example.json", r'{"path": "\\\\server\\share\\model"}', "UNC absolute path"),
        ("data/example.json", r'{"cwd": "\\\\server\\share\\model"}', "UNC absolute path"),
        ("data/example.json", r'{"path": "D:\\new\\model"}', "JSON-escaped Windows absolute path"),
        ("config.yaml", "ip_address: 10.23.45.67", "private network address"),
        ("config.yaml", "MASTER_ADDR=10.23.45.67", "private network address"),
        ("config.yaml", "endpoint: http://10.23.45.67:8080", "private network address"),
        ("config.yaml", "HostName compute.example", "SSH host name"),
        ("config.yaml", "host: worker.example", "recorded host name"),
        ("docs/example.md", "ssh user@compute.example", "remote host in shell command"),
        ("report.json", f"device_uuid: {gpu_id}", "GPU hardware UUID"),
        ("reports/example.log", "/custom/team/repo/file", "absolute path in public report"),
    ]
    negatives = [
        ("docs/example.md", "${PROJECT_ROOT}/data/train.jsonl"),
        ("docs/example.md", "https://example.com/data/train.jsonl"),
        ("config.yaml", "host: localhost"),
        ("config.yaml", "version: 10.3.10.19"),
        ("data/example.json", r'{"answer": "A:\nnext"}'),
        ("data/example.json", r'{"solution": "\\\\alpha\\beta"}'),
        ("report.json", f'{{"request_id": "{gpu_id}"}}'),
    ]
    missing = [
        label
        for relative, line, label in positives
        if label not in violation_labels(relative, line)
    ]
    false_positives = [
        (relative, line)
        for relative, line in negatives
        if violation_labels(relative, line)
    ]
    if missing or false_positives:
        raise RuntimeError(
            f"Portability rule self-check failed: missing={missing}, "
            f"false_positives={false_positives}"
        )


def tracked_files():
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode()
        if relative == SELF or relative.startswith(SKIPPED_PREFIXES):
            continue
        yield relative, ROOT / relative


def main():
    verify_rules()
    findings = []
    checked = 0
    for relative, path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        for line_number, line in enumerate(text.splitlines(), 1):
            for label in violation_labels(relative, line):
                findings.append(f"{relative}:{line_number}: {label}")
    if findings:
        print("\n".join(findings))
        raise SystemExit(f"Found {len(findings)} portability violation(s).")
    print(f"Portable path check passed for {checked} tracked text files.")


if __name__ == "__main__":
    main()
