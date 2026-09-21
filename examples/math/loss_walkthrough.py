#!/usr/bin/env python3
"""CPU-only numerical companion to the rewritten loss tutorial.

No model download, environment simulator, repository package, or network is used.
These are constructed mathematical examples, not MM-RL-Learning training results.
Run: python loss_walkthrough.py --output numerical_checks.json
Dependency: PyTorch. Tested version is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("This example requires PyTorch in the active Python environment.") from exc

DTYPE = torch.float64


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def ppo_term(logp: torch.Tensor, old_logp: torch.Tensor,
             advantage: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """Single-sample loss; no value loss, KL, entropy, or batch reduction."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    ratio = (logp - old_logp.detach()).exp()
    a = advantage.detach()
    return -torch.minimum(ratio * a, ratio.clamp(1 - epsilon, 1 + epsilon) * a)


def safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Demonstrate an explicit contract: finite active entries; nonempty mask.

    This cleans already-created leaf values. It does NOT repair an earlier
    invalid differentiable operation: mask operands before log/exp/division too.
    """
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not bool(mask.any()):
        raise ValueError("An empty mask is a skipped sample, not a learning signal")
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError("Active entries contain NaN or infinity")
    clean = torch.where(mask, values, torch.zeros_like(values))
    return clean.sum() / mask.sum()


def run_checks() -> dict[str, Any]:
    results: dict[str, Any] = {
        "scope": "Constructed CPU mathematical checks only; not repository training or its full test suite",
        "torch_version": torch.__version__,
        "dtype": "float64", "device": "cpu",
    }

    # 1. Cross entropy receives logits, not probabilities passed through softmax twice.
    logits = torch.tensor([[math.log(0.6), math.log(0.3), math.log(0.1)]],
                          dtype=DTYPE, requires_grad=True)
    ce = F.cross_entropy(logits, torch.tensor([0]))
    ce.backward()
    close(float(ce.detach()), -math.log(0.6), "CE")
    for got, expected in zip(logits.grad[0].tolist(), [-0.4, 0.3, 0.1]):
        close(got, expected, "CE gradient")
    results["cross_entropy"] = {"loss": float(ce.detach()), "logit_gradient": logits.grad[0].tolist()}

    # 2. A scalar value prediction is regressed toward a fixed return target.
    value = torch.tensor(0.3, dtype=DTYPE, requires_grad=True)
    target = torch.tensor(1.0, dtype=DTYPE)
    mse = (value - target).square()
    mse.backward()
    close(float(mse.detach()), 0.49, "MSE")
    close(float(value.grad), -1.4, "MSE gradient")
    results["value_regression"] = {"loss": float(mse.detach()), "gradient": float(value.grad)}

    # 3. With binary target 1, BCE on a margin equals -log sigmoid(margin).
    margin = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    bce = F.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
    bce.backward()
    close(float(bce.detach()), math.log(2), "BCE")
    close(float(margin.grad), -0.5, "BCE gradient")
    results["pairwise_logistic"] = {"loss": float(bce.detach()), "margin_gradient": float(margin.grad)}

    # 4. One fixed, selected action receives a positive advantage. Other logits move too.
    parameters = torch.tensor([math.log(0.2), math.log(0.8)], dtype=DTYPE, requires_grad=True)
    before = parameters.softmax(-1).detach()
    selected_logp = parameters.log_softmax(-1)[0]
    policy_loss = -selected_logp  # advantage = +1
    policy_loss.backward()
    grad = parameters.grad.detach().clone()
    close(float(grad[0]), -0.8, "PG selected-logit gradient")
    close(float(grad[1]), 0.8, "PG other-logit gradient")
    with torch.no_grad():
        parameters -= 0.1 * parameters.grad
    after = parameters.softmax(-1).detach()
    if not float(after[0]) > float(before[0]):
        raise AssertionError("The selected action probability should increase in this one-sample example")
    results["one_policy_update"] = {"probabilities_before": before.tolist(),
                                    "logit_gradient": grad.tolist(), "probabilities_after": after.tolist()}

    # 5. Ratio equals one at the snapshot, but d ratio / d current_logp = one.
    current = torch.tensor(math.log(0.2), dtype=DTYPE, requires_grad=True)
    old = current.detach().clone()
    ratio = (current - old).exp()
    ratio.backward()
    close(float(ratio.detach()), 1.0, "snapshot ratio")
    close(float(current.grad), 1.0, "snapshot derivative")
    wrong = torch.tensor(math.log(0.2), dtype=DTYPE, requires_grad=True)
    wrong_ratio = (wrong - wrong).exp()
    wrong_ratio.backward()
    close(float(wrong.grad), 0.0, "shared-graph cancellation")
    results["old_snapshot"] = {"ratio": float(ratio.detach()), "current_logp_gradient": float(current.grad),
                               "wrong_shared_graph_gradient": float(wrong.grad)}

    # 6. Four quadrants; stay away from the nondifferentiable clipping boundaries.
    clipping_rows = []
    cases = [(1.0, 0.5, -0.5, -0.5), (1.0, 1.5, -1.2, 0.0),
             (-1.0, 0.5, 0.8, 0.0), (-1.0, 1.5, 1.5, 1.5)]
    for advantage, ratio_value, expected_loss, expected_gradient in cases:
        logp = torch.tensor(math.log(0.2 * ratio_value), dtype=DTYPE, requires_grad=True)
        loss = ppo_term(logp, torch.tensor(math.log(0.2), dtype=DTYPE),
                        torch.tensor(advantage, dtype=DTYPE))
        loss.backward()
        close(float(loss.detach()), expected_loss, "PPO loss")
        close(float(logp.grad), expected_gradient, "PPO logp gradient")
        clipping_rows.append({"advantage": advantage, "ratio": ratio_value,
                              "loss": float(loss.detach()), "d_loss_d_logp": float(logp.grad)})
    results["ppo_four_cases"] = clipping_rows

    # 7. Symmetric, detached-weight teaching example (not the full original CISPO recipe).
    logp = torch.tensor(math.log(0.3), dtype=DTYPE, requires_grad=True)
    ratio = (logp - math.log(0.2)).exp()
    cispo_teaching = -ratio.clamp(0.8, 1.2).detach() * logp
    cispo_teaching.backward()
    close(float(logp.grad), -1.2, "detached-weight derivative")
    results["cispo_teaching_variant"] = {"ratio": float(ratio.detach()), "d_loss_d_logp": float(logp.grad)}

    # 8. Masked infinity is a helper-level edge case, not proof of a repository pipeline failure.
    values = torch.tensor([1.0, -float("inf")], dtype=DTYPE, requires_grad=True)
    mask = torch.tensor([True, False])
    unsafe = (values * mask).sum() / mask.sum()
    if not bool(torch.isnan(unsafe)):
        raise AssertionError("The constructed 0 * -inf case should be NaN")
    safe = safe_masked_mean(values, mask)
    safe.backward()
    close(float(safe.detach()), 1.0, "safe masked mean")
    close(float(values.grad[0]), 1.0, "active gradient")
    close(float(values.grad[1]), 0.0, "inactive direct gradient")
    results["masked_nonfinite_leaf"] = {"naive_mean_is_nan": True, "safe_mean": float(safe.detach()),
                                        "direct_leaf_gradient": values.grad.tolist()}

    # 9. Different reductions encode different sample weights.
    token_losses = torch.tensor([[2., 2., 0., 0., 0., 0., 0., 0.],
                                 [1., 1., 1., 1., 1., 1., 1., 1.]], dtype=DTYPE)
    mask = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1., 1., 1.]], dtype=DTYPE)
    sequence_mean = ((token_losses * mask).sum(-1) / mask.sum(-1)).mean()
    token_mean = (token_losses * mask).sum() / mask.sum()
    fixed_length = (token_losses * mask).sum() / (2 * 8)
    close(float(sequence_mean), 1.5, "sequence mean")
    close(float(token_mean), 1.2, "token mean")
    close(float(fixed_length), 0.75, "fixed-length mean")
    results["reductions"] = {"sequence_mean": float(sequence_mean), "token_mean": float(token_mean),
                             "fixed_length_mean": float(fixed_length)}

    # 10. A single sample's log-ratio can be negative although full KL is nonnegative.
    p = torch.tensor([0.8, 0.2], dtype=DTYPE)
    q = torch.tensor([0.5, 0.5], dtype=DTYPE)
    sample_terms = (p / q).log()
    kl = (p * sample_terms).sum()
    if not float(sample_terms[1]) < 0 < float(kl):
        raise AssertionError("KL example has the wrong sign")
    results["kl"] = {"sample_log_ratios": sample_terms.tolist(), "full_distribution_kl": float(kl)}
    results["status"] = "passed"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON result path; refuses to overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}. Choose a new path.")
    results = run_checks()
    rendered = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
