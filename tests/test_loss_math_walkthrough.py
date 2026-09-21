"""L0 companion checks for the rewritten loss tutorial."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = ROOT / "examples" / "math" / "loss_walkthrough.py"


def _load_walkthrough():
    spec = importlib.util.spec_from_file_location("loss_walkthrough", WALKTHROUGH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_checks_status():
    payload = json.loads(
        (ROOT / "examples/math/reference/numerical_checks.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "passed"
    assert math.isclose(payload["cross_entropy"]["loss"], -math.log(0.6), rel_tol=1e-9)
    assert math.isclose(payload["value_regression"]["gradient"], -1.4, rel_tol=1e-9)
    assert len(payload["ppo_four_cases"]) == 4


def test_walkthrough_module_runs_when_torch_available():
    pytest.importorskip("torch")
    module = _load_walkthrough()
    results = module.run_checks()
    assert results["status"] == "passed"
    assert results["old_snapshot"]["ratio"] == pytest.approx(1.0)
    assert results["old_snapshot"]["current_logp_gradient"] == pytest.approx(1.0)
    assert results["old_snapshot"]["wrong_shared_graph_gradient"] == pytest.approx(0.0)
    cases = results["ppo_four_cases"]
    assert cases[1]["d_loss_d_logp"] == pytest.approx(0.0)
    assert cases[3]["d_loss_d_logp"] == pytest.approx(1.5)


def test_chapters_registry_has_learning_metadata():
    chapters = json.loads((ROOT / "configs/chapters.json").read_text(encoding="utf-8"))
    assert chapters
    by_id = {row.get("id"): row for row in chapters if row.get("id")}
    assert "loss-basics" in by_id
    assert by_id["grpo"]["prerequisites"] == ["loss-basics"]
    assert "ppo" in by_id["grpo"].get("recommended_prerequisites", [])


def test_policy_loss_rejects_unknown_kind():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "agentic_rl_losses",
        ROOT / "src/agentic_rl/losses.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch = pytest.importorskip("torch")
    logp = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    adv = torch.ones(2, 3)
    mask = torch.ones(2, 3)
    with pytest.raises(ValueError, match="Unknown policy loss kind"):
        module.policy_loss(logp, old, adv, mask, kind="not-a-real-algorithm")


