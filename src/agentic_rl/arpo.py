"""ARPO: Agentic Reinforced Policy Optimization (teaching core).

Source: Dong et al., Agentic Reinforced Policy Optimization, arXiv:2507.19849
(ICLR 2026). Code reference: https://github.com/RUC-NLPIR/ARPO

Local teaching subset (not the full upstream verl stack):
- entropy pre-monitor after tool feedback
- adaptive branch sampling on high-uncertainty tool-call steps
- step-level advantage attribution for tool-use credit
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ArpoRolloutConfig:
    initial_rollouts: int = 8
    beam_size: int = 2
    branch_probability: float = 0.5
    entropy_weight: float = 0.2
    entropy_threshold: float = 1.0
    max_branches_per_turn: int = 2


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits) over the last dim. Shape [..., V] -> [...]"""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def should_branch_after_tool(
    post_tool_entropy: float,
    *,
    cfg: ArpoRolloutConfig,
    already_branched: int = 0,
    rng: torch.Generator | None = None,
) -> bool:
    """ARPO branch decision: high entropy after tool feedback => optional extra beam."""
    if already_branched >= cfg.max_branches_per_turn:
        return False
    if post_tool_entropy < cfg.entropy_threshold:
        return False
    u = torch.rand((), generator=rng).item()
    return u < cfg.branch_probability


def allocate_branch_budget(
    entropies: list[float],
    *,
    cfg: ArpoRolloutConfig,
    total_budget: int | None = None,
) -> list[int]:
    """How many rollouts to spend at each step (global + branch).

    entropies[i] is the post-tool entropy at step i. Higher entropy gets more
    of the branch budget, capped by beam_size * max_branches.
    """
    budget = total_budget if total_budget is not None else cfg.initial_rollouts
    if not entropies:
        return []
    base = [1] * len(entropies)
    remaining = max(0, budget - len(entropies))
    if remaining == 0:
        return base
    weights = [max(0.0, e - cfg.entropy_threshold) + 1e-6 for e in entropies]
    total_w = sum(weights)
    extra = [int(remaining * w / total_w) for w in weights]
    for i in range(len(entropies)):
        cap = cfg.beam_size * cfg.max_branches_per_turn
        extra[i] = min(extra[i], cap)
        base[i] += extra[i]
    return base


def attribute_step_advantages(
    trajectory_rewards: torch.Tensor,
    step_rewards: torch.Tensor | None = None,
    *,
    baseline: float | None = None,
    entropy: torch.Tensor | None = None,
    entropy_weight: float = 0.2,
) -> torch.Tensor:
    """Advantage attribution over tool steps.

    trajectory_rewards: [B] terminal/group reward for each rollout
    step_rewards: optional [B, T] sparse step feedback (tool success etc.)
    If step_rewards is given, redistribute trajectory advantage across steps
    proportional to step signal + small uniform mass; else broadcast equally.

    entropy: optional [B, T]; adds entropy_weight * normalized entropy to the
    attribution weights (encourage learning on uncertain tool steps).
    """
    if trajectory_rewards.ndim != 1:
        raise ValueError("trajectory_rewards must be [B]")
    if step_rewards is None:
        b = trajectory_rewards.shape[0]
        return trajectory_rewards.unsqueeze(-1).expand(b, 1).clone()
    if step_rewards.shape[0] != trajectory_rewards.shape[0]:
        raise ValueError("batch mismatch between rewards and step_rewards")
    # Group-relative baseline on trajectory rewards if not given.
    if baseline is None:
        baseline = float(trajectory_rewards.mean())
    traj_adv = trajectory_rewards - baseline  # [B]
    weights = step_rewards.abs() + 1e-3  # [B, T]
    if entropy is not None:
        if entropy.shape != step_rewards.shape:
            raise ValueError("entropy must match step_rewards shape")
        # normalize per-row entropy into weights
        e = entropy - entropy.amin(dim=-1, keepdim=True)
        denom = e.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights + entropy_weight * (e / denom)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return traj_adv.unsqueeze(-1) * weights  # [B, T]


def arpo_surrogate_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> torch.Tensor:
    """Clipped PG on already-attributed step/token advantages.

    logp/old_logp/advantages/mask: [B, T]
    """
    from .losses import policy_loss

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    loss, _ = policy_loss(
        logp,
        old_logp,
        advantages,
        mask,
        kind="clipped_pg",
        clip_low=clip_low,
        clip_high=clip_high,
        reduction="token_mean",
    )
    return loss


def demo() -> dict:
    """CPU numeric demo used by smoke tests and tutorials."""
    cfg = ArpoRolloutConfig()
    ents = [0.4, 1.8, 2.2]
    budget = allocate_branch_budget(ents, cfg=cfg, total_budget=10)
    flags = [
        should_branch_after_tool(e, cfg=cfg, already_branched=0, rng=torch.Generator().manual_seed(0))
        for e in ents
    ]
    # two rollouts, three tool steps
    traj = torch.tensor([1.0, 0.0])
    steps = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ent = torch.tensor([[0.2, 2.0, 0.1], [1.5, 0.3, 1.8]])
    adv = attribute_step_advantages(traj, steps, entropy=ent, entropy_weight=0.2)
    return {
        "branch_budget": budget,
        "branch_flags": flags,
        "step_advantages": adv.tolist(),
        "expected_budget_sum": sum(budget),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
