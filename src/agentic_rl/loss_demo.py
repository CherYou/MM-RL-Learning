"""Generate objective and gradient curves with a reproducible CPU tensor example."""

import csv
import json
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ROOT, report_path
from .losses import policy_loss
from .logging import RunLogger


def run():
    import time

    path = ROOT / f"runs/loss-functions-{time.time_ns()}"
    logger = RunLogger(path, {"algorithm": "loss-functions", "device": "cpu"})
    ratios = torch.linspace(0.5, 1.5, 101)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = []
    for sign, ax in zip((1.0, -1.0), axes):
        for kind in ("importance_sampling", "grpo", "cispo", "gspo"):
            losses = []
            grads = []
            for ratio in ratios:
                lp = ratio.log().reshape(1, 1).requires_grad_()
                loss, _ = policy_loss(
                    lp, torch.zeros_like(lp), torch.tensor([sign]), torch.ones_like(lp), kind=kind
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
            xlabel="policy / rollout probability",
            ylabel="d loss / d log probability",
        )
        ax.axvline(0.8, color="grey", linestyle=":")
        ax.axvline(1.2, color="grey", linestyle=":")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path / "gradients.png", dpi=160)
    plt.close(fig)
    with (path / "curves.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for i, r in enumerate(rows[:101]):
        logger.log(i, {"ratio": r["ratio"], "gradient": r["d_loss_d_logp"]})
    logger.close()
    (path / "status.json").write_text(json.dumps({"status": "completed", "device": "cpu"}) + "\n")
    print(report_path(path))
    return path
