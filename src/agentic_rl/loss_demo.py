"""Generate objective and gradient curves with a reproducible CPU tensor example."""

import csv
import json
import math
import time

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import ROOT, report_path
from .logging import RunLogger
from .losses import policy_loss


def run():
    path = ROOT / f"runs/loss-functions-{time.time_ns()}"
    logger = RunLogger(path, {"algorithm": "loss-functions", "device": "cpu"})
    ratios = torch.linspace(0.5, 1.5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    p_old = 0.2
    old_logp = math.log(p_old)
    try:
        for sign, ax in zip((1.0, -1.0), axes):
            for kind in ("importance_sampling", "grpo", "cispo", "gspo"):
                sign_name = "positive" if sign > 0 else "negative"
                prefix = f"loss_demo/{kind}/{sign_name}"
                for ratio_index, ratio in enumerate(ratios):
                    lp = torch.tensor(
                        math.log(p_old * float(ratio)), dtype=torch.float64
                    ).reshape(1, 1)
                    lp.requires_grad_()
                    loss, _ = policy_loss(
                        lp,
                        torch.tensor(old_logp, dtype=torch.float64).reshape(1, 1),
                        torch.tensor([sign], dtype=torch.float64),
                        torch.ones_like(lp),
                        kind=kind,
                    )
                    loss.backward()
                    rows.append(
                        {
                            "advantage": sign,
                            "kind": kind,
                            "ratio": float(ratio),
                            "loss": float(loss.detach()),
                            "d_loss_d_logp": float(lp.grad),
                        }
                    )
                    logger.log(
                        ratio_index,
                        {
                            f"{prefix}/ratio": float(ratio),
                            f"{prefix}/loss": float(loss.detach()),
                            f"{prefix}/d_loss_d_logp": float(lp.grad),
                        },
                    )
                ax.plot(
                    ratios,
                    [r["d_loss_d_logp"] for r in rows if r["kind"] == kind and r["advantage"] == sign],
                    label=kind,
                )
            ax.set(
                title=f"Advantage = {sign:g}",
                xlabel="policy / rollout probability ratio",
                ylabel="d loss / d log probability",
            )
            ax.axvline(0.8, color="grey", linestyle=":")
            ax.axvline(1.2, color="grey", linestyle=":")
            ax.legend()

        # Multi-token contrast for GSPO vs GRPO-style aggregation (same clip for teaching).
        token_ratios = torch.tensor([2.0, 0.5], dtype=torch.float64)
        geometric = token_ratios.log().mean().exp()
        arithmetic = float(token_ratios.mean())
        # Teaching losses with A=1, clip [0.8,1.2]: GRPO token-wise then mean; GSPO geom then clip.
        grpo_loss = -float(
            (
                torch.minimum(token_ratios, token_ratios.clamp(0.8, 1.2))
            ).mean()
        )
        gspo_loss = -float(min(float(geometric), 1.2))
        for kind_name, ratio_val, loss_val in (
            ("gspo_geometric_mean_demo", float(geometric), gspo_loss),
            ("arithmetic_mean_reference", arithmetic, None),
            ("grpo_tokenwise_mean_demo", None, grpo_loss),
        ):
            rows.append(
                {
                    "advantage": 1.0 if loss_val is not None else float("nan"),
                    "kind": kind_name,
                    "ratio": ratio_val if ratio_val is not None else float("nan"),
                    "loss": loss_val if loss_val is not None else float("nan"),
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
        (path / "status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "device": "cpu",
                    "p_old": p_old,
                    "gspo_token_ratios": [2.0, 0.5],
                    "gspo_geometric_mean": float(geometric),
                    "arithmetic_mean_reference": arithmetic,
                    "grpo_tokenwise_mean_demo_loss": grpo_loss,
                    "gspo_geometric_demo_loss": gspo_loss,
                    "note": "arithmetic_mean_reference is not GRPO's loss definition",
                },
                allow_nan=False,
            )
            + "\n"
        )
    finally:
        logger.close()
    print(report_path(path))
    return path
