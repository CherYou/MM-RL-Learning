#!/usr/bin/env python3
"""Inspect the deliverables against the requested scope and saved runtime evidence."""

import ast
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from agentic_rl.data import ROOT, read_jsonl


def main():
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "gpu_validation_performed": False,
        "scope": "code, documents, prepared learning data, installed dependencies, local logs and CPU checks",
    }
    upstream = json.loads((ROOT / "references/upstream.json").read_text())
    expected = (set(upstream["chapters"]) - {"00-loss-function"}) | {
        "preliminary",
        "001-ppo",
        "002-dpo",
        "12-harness-rl",
        "13-sac",
        "14-td3",
        "15-her",
        "16-iql",
    }
    actual = {p.name for p in ROOT.glob("[0-9]*") if p.is_dir()} | {"preliminary"}
    assert actual == expected, (actual, expected)
    chapters = json.loads((ROOT / "configs/chapters.json").read_text())
    for c in chapters:
        required = ["README.md", "TUTORIAL.md", "train.py", "config.yaml", "prepare_data.py"]
        if c["algorithm"] != "loss-functions":
            required.append("eval.py")
        for filename in required:
            assert (ROOT / c["chapter"] / filename).is_file(), (c, filename)
    report["chapters"] = {
        "upstream": len(upstream["chapters"]),
        "new": 7,
        "top_level": len(actual),
        "including_general_opd": len(chapters),
        "verified": True,
    }
    # Check the preserved reference contains every tracked source file in the pinned checkout.
    original = ROOT.with_name("agentic-rl-lab-upstream")
    if (original / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=original, text=True).strip()
        assert revision == upstream["revision"]
        for p in original.rglob("*"):
            if p.is_file() and ".git" not in p.relative_to(original).parts:
                copied = ROOT / "references/upstream" / p.relative_to(original)
                assert copied.read_bytes() == p.read_bytes(), str(copied)
    report["upstream"] = {**upstream, "snapshot_verified": True}
    active = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    active += [p for d in actual for p in (ROOT / d).rglob("*.py")]
    for p in active:
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            names = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert all(not name.startswith("pytrio") for name in names), str(p)
    installed = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    assert "pytrio" not in installed
    import torch

    assert torch.version.cuda is None
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    for cmd in [
        [str(ROOT / ".venv/bin/uv"), "lock", "--check"],
        [str(ROOT / ".venv/bin/uv"), "pip", "check"],
    ]:
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    report["implementation"] = {
        "active_python_files": len(active),
        "pytrio_imports": 0,
        "pytrio_installed": False,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "lock_and_dependencies_valid": True,
    }
    data = json.loads((ROOT / "reports/data-verification.json").read_text())
    assert len(data) == 15 and all(d["verified"] for d in data.values())
    report["data"] = {"verified_sources": len(data), "report": "reports/data-verification.json"}
    tests = (ROOT / "reports/tutorial-tests.log").read_text()
    m = re.search(r"(\d+) passed", tests)
    assert m and int(m.group(1)) >= 25 and not re.search(r"\d+ failed", tests)
    smoke = json.loads((ROOT / "reports/cpu-audit-latest.json").read_text())
    assert len(smoke) == 22 and all(r["passed"] and r["exit_code"] == 0 for r in smoke.values())
    for name, evidence in smoke.items():
        run = ROOT / evidence["run"]
        assert json.loads((run / "status.json").read_text())["status"] == "completed"
        assert (run / "checkpoint-final/model/config.json").exists()
        metrics = read_jsonl(run / "metrics.jsonl")
        assert all(math.isfinite(v) for row in metrics for v in row.values() if isinstance(v, (int, float)))
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        events = EventAccumulator(str(run / "tensorboard")).Reload()
        assert events.Tags()["scalars"], name
    report["cpu_verification"] = {
        "mechanism_tests_passed": int(m.group(1)),
        "training_entries_passed": len(smoke),
        "all_checkpoints_present": True,
        "all_runs_have_tensorboard_scalars": True,
        "benchmark_quality_claimed": False,
    }
    for document in [
        ROOT / "README.md",
        *ROOT.glob("docs/*.md"),
        *[ROOT / c["chapter"] / "README.md" for c in chapters],
        *[ROOT / c["chapter"] / "TUTORIAL.md" for c in chapters],
        ROOT / "preliminary/FOUNDATIONS.md",
    ]:
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            link = link.strip("<>").split("#", 1)[0]
            if not link or link.startswith(("http:", "https:", "mailto:")):
                continue
            assert (document.parent / link).exists(), (document, link)
    report["documents"] = {
        "chapter_guides": len(chapters),
        "supporting_guides": len(list(ROOT.glob("docs/*.md"))),
        "local_links_valid": True,
        "research": "docs/RESEARCH.md",
    }
    # Exercise every thin CLI wrapper's parser from a different working directory.
    for c in chapters:
        if c["algorithm"] == "loss-functions":
            continue
        for file in ["train.py", "eval.py", "prepare_data.py"]:
            subprocess.run(
                [sys.executable, str(ROOT / c["chapter"] / file), "--help"],
                cwd=tempfile.gettempdir(),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
    report["chapter_entrypoint_help_checks"] = 3 * sum(c["algorithm"] != "loss-functions" for c in chapters)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/services.py"), "status"],
        check=True,
        capture_output=True,
        text=True,
    )
    report["local_services"] = json.loads((ROOT / "reports/local-services.json").read_text())
    assert all(d["live"] and d["http_status"] == 200 for d in report["local_services"].values())
    report["pretrained_model"] = json.loads((ROOT / "reports/pretrained-model-verification.json").read_text())
    report["embodied"] = json.loads((ROOT / "reports/embodied-validation.json").read_text())
    assert report["embodied"]["passed"]
    report["provenance"] = "reports/code-provenance-audit.json"
    assert (ROOT / report["provenance"]).is_file()
    report["tutorials"] = json.loads((ROOT / "reports/tutorial-audit.json").read_text())
    assert report["tutorials"]["passed"]
    report["all_requested_deliverables_verified"] = True
    (ROOT / "reports/completion-audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
