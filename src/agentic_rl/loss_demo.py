"""Generate objective and gradient curves with a reproducible CPU tensor example."""

import csv
import json

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import ROOT, report_path
from .logging import RunLogger
from .losses import policy_loss


def run():
    import math
    import time

    path = ROOT / f"runs/loss-functions-{time.time_ns()}"
    logger = RunLogger(path, {"algorithm": "loss-functions", "device": "cpu"})
    ratios = torch.linspace(0.5, 1.5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    p_old = 0.2
    old_logp = math.log(p_old)
    for sign, ax in zip((1.0, -1.0), axes):
        for kind in ("importance_sampling", "grpo", "cispo", "gspo"):
            losses = []
            grads = []
            for ratio in ratios:
                # Legal discrete probabilities: p = p_old * ratio, then take log.
                lp = torch.tensor(math.log(p_old * float(ratio)), dtype=torch.float64).reshape(1, 1)
                lp.requires_grad_()
                loss, _ = policy_loss(
                    lp,
                    torch.tensor(old_logp, dtype=torch.float64).reshape(1, 1),
                    torch.tensor([sign], dtype=torch.float64),
                    torch.ones_like(lp),
                    kind=kind,
                )
                loss.backward()
                losses.append(float(loss.detach()))
                grads.append(float(lp.grad))
                rows.append(
                    {
                        "advantage": sign,
                        "kind": kind,
                        "ratio": float(ratio),
                        "loss": float(loss.detach()),
                        "d_loss_d_logp": float(lp.grad),
                    }
                )
            ax.plot(ratios, grads, label=kind)
        ax.set(
            title=f"Advantage = {sign:g}",
            xlabel="policy / rollout probability ratio",
            ylabel="d loss / d log probability",
        )
        ax.axvline(0.8, color="grey", linestyle=":")
        ax.axvline(1.2, color="grey", linestyle=":")
        ax.legend()
    # Multi-token contrast: geometric mean vs arithmetic mean for GSPO-style ratios.
    token_ratios = torch.tensor([2.0, 0.5], dtype=torch.float64)
    geometric = token_ratios.log().mean().exp()
    arithmetic = token_ratios.mean()
    rows.append(
        {
            "advantage": float("nan"),
            "kind": "gspo_geometric_mean_demo",
            "ratio": float(geometric),
            "loss": float("nan"),
            "d_loss_d_logp": float("nan"),
        }
    )
    rows.append(
        {
            "advantage": float("nan"),
            "kind": "grpo_arithmetic_mean_demo",
            "ratio": float(arithmetic),
            "loss": float("nan"),
            "d_loss_d_logp": float("nan"),
        }
    )
    fig.tight_layout()
    fig.savefig(path / "gradients.png", dpi=160)
    plt.close(fig)
    with (path / "curves.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Log every algorithm/advantage combination instead of only the first positive-IS slice.
    for i, r in enumerate(rows):
        if any(isinstance(r[k], float) and math.isnan(r[k]) for k in ("loss", "d_loss_d_logp")):
            continue
        payload = {
            "advantage": r["advantage"],
            "kind": r["kind"],
            "ratio": r["ratio"],
            "loss": r["loss"],
            "gradient": r["d_loss_d_logp"],
        }
        logger.log(i, payload)
    logger.close()
    (path / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "device": "cpu",
                "p_old": p_old,
                "gspo_token_ratios": [2.0, 0.5],
                "gspo_geometric_mean": float(geometric),
                "grpo_arithmetic_mean": float(arithmetic),
            },
            allow_nan=False,
        )
        + "\n"
    )
    print(report_path(path))
    return path
