"""AEPO: Agentic Entropy-Balanced Policy Optimization (teaching core).

Source: Dong et al., Agentic Entropy-Balanced Policy Optimization, arXiv:2510.14545
(WWW 2026 Oral). Code: https://github.com/RUC-NLPIR/ARPO/tree/main/AEPO

Local teaching subset:
- dynamic entropy-balanced rollout budget (+ branch penalty)
- entropy clipping-balanced (stop-grad on high-entropy clip weight)
- entropy-aware advantage estimation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AepoConfig:
    enable_dynamic_rollouts: bool = True
    enable_entropy_balanced_clipping: bool = True
    enable_entropy_balanced_advantage: bool = True
    entropy_high: float = 1.0
    branch_penalty: float = 0.1
    clip_low: float = 0.2
    clip_high: float = 0.2
    entropy_adv_weight: float = 0.2


def dynamic_entropy_budget(
    entropies: torch.Tensor,
    *,
    cfg: AepoConfig,
    base_budget: int,
    max_branch: int,
) -> torch.Tensor:
    """Allocate extra branches where entropy is high; penalize consecutive high-entropy turns.

    entropies: [T] post-tool entropies along one trajectory (or mean over batch).
    Returns integer budget per step, shape [T].
    """
    if entropies.ndim != 1:
        raise ValueError("entropies must be 1-D")
    t = entropies.numel()
    budget = torch.ones(t, dtype=torch.long)
    if not cfg.enable_dynamic_rollouts:
        return budget * max(1, base_budget // max(1, t))
    high = entropies > cfg.entropy_high
    extra = torch.zeros(t)
    extra[high] = 1.0
    # consecutive high-entropy tool calls => branch penalty
    for i in range(1, t):
        if bool(high[i]) and bool(high[i - 1]):
            extra[i] = max(0.0, float(extra[i]) - cfg.branch_penalty * i)
    extra = extra / extra.sum().clamp_min(1.0)
    alloc = (extra * (base_budget - t)).clamp_min(0)
    budget = budget + alloc.round().long()
    return budget.clamp_max(max_branch)


def entropy_balanced_clip(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    logp: torch.Tensor,
    entropy: torch.Tensor,
    *,
    cfg: AepoConfig,
) -> torch.Tensor:
    """Teaching form of AEPO entropy clipping-balanced surrogate (per token).

    For high-entropy tokens, the clipped weight is stop-grad so the residual
    logp gradient is preserved and rescaled instead of zeroed by PPO clip.

    ratio, advantage, logp, entropy: [...]
    """
    if not (ratio.shape == advantage.shape == logp.shape == entropy.shape):
        raise ValueError("shape mismatch")
    eps_l, eps_h = cfg.clip_low, cfg.clip_high
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps_l, 1 + eps_h) * advantage
    chosen = torch.minimum(unclipped, clipped)
    if cfg.enable_entropy_balanced_clipping:
        high = entropy > cfg.entropy_high
        # stop-grad clip weight on high-entropy positions
        weight = torch.where(
            high,
            ratio.clamp(1 - eps_l, 1 + eps_h).detach(),
            torch.ones_like(ratio),
        )
        # keep a path through logp on high-entropy tokens
        chosen = torch.where(high, weight * advantage * logp, chosen)
    return -chosen


def entropy_aware_advantage(
    advantage: torch.Tensor,
    entropy: torch.Tensor,
    *,
    cfg: AepoConfig,
) -> torch.Tensor:
    """Rescale advantages toward high-uncertainty tool tokens.

    advantage, entropy: [B, T]
    """
    if not cfg.enable_entropy_balanced_advantage:
        return advantage
    e = entropy - entropy.mean(dim=-1, keepdim=True)
    scale = 1.0 + cfg.entropy_adv_weight * torch.tanh(e)
    return advantage * scale


def demo() -> dict:
    cfg = AepoConfig()
    ents = torch.tensor([0.3, 1.4, 2.1, 2.3])
    budget = dynamic_entropy_budget(ents, cfg=cfg, base_budget=10, max_branch=4)
    ratio = torch.tensor([1.5, 1.5])
    adv = torch.tensor([1.0, -1.0])
    logp = torch.tensor([-1.0, -1.2])
    ent = torch.tensor([2.0, 0.2])  # first is high-entropy tool token
    loss_tok = entropy_balanced_clip(ratio, adv, logp, ent, cfg=cfg)
    adv2 = entropy_aware_advantage(
        torch.tensor([[1.0, -1.0]]), torch.tensor([[2.0, 0.1]]), cfg=cfg
    )
    return {
        "dynamic_budget": budget.tolist(),
        "high_entropy_token_loss": float(loss_tok[0]),
        "low_entropy_token_loss": float(loss_tok[1]),
        "entropy_aware_advantage": adv2.tolist(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(demo(), indent=2))
