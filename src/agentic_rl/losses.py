"""Explicit PyTorch objectives. All arrays have shape [batch, response positions].

Old/reference/teacher probabilities are constants. Masks exclude prompts, padding,
tool observations and replayed context. A zero-variance reward group stays zero.
"""

import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def group_advantages(rewards, group_size, scale=True):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("Every prompt must have a complete group of at least two rollouts")
    groups = rewards.reshape(-1, group_size)
    centered = groups - groups.mean(-1, keepdim=True)
    if scale:
        centered = centered / (groups.std(-1, keepdim=True, correction=0) + 1e-4)
    return centered.flatten().detach()


ALLOWED_POLICY_LOSS_KINDS = {
    "grpo",
    "ppo",
    "clipped_pg",
    "importance_sampling",
    "cispo",
    "gspo",
    "dapo",
    "dr_grpo",
}


def policy_loss(
    logp,
    old_logp,
    advantage,
    mask,
    *,
    kind="grpo",
    clip_low=0.2,
    clip_high=0.2,
    ref_logp=None,
    beta=0.0,
    max_length=None,
    reduction="auto",
):
    if kind not in ALLOWED_POLICY_LOSS_KINDS:
        raise ValueError(
            f"Unknown policy loss kind: {kind!r}. Allowed: {sorted(ALLOWED_POLICY_LOSS_KINDS)}"
        )
    if reduction not in {"auto", "token_mean", "sequence_mean", "fixed_length"}:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    old_logp = old_logp.detach()
    if advantage.ndim == 1:
        advantage = advantage[:, None]
    advantage = advantage.detach()
    delta = logp - old_logp
    if kind == "gspo":
        delta = (delta * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    ratio = delta.exp()
    if kind == "importance_sampling":
        objective = ratio * advantage
    elif kind == "cispo":
        objective = ratio.clamp(1 - clip_low, 1 + clip_high).detach() * advantage * logp
    else:
        objective = torch.minimum(
            ratio * advantage,
            ratio.clamp(1 - clip_low, 1 + clip_high) * advantage,
        )
    per_token = -objective
    kl = torch.zeros_like(logp)
    if beta:
        if ref_logp is None:
            raise ValueError("A fixed reference policy is required for nonzero KL beta")
        gap = ref_logp.detach() - logp
        kl = gap.exp() - gap - 1
        per_token = per_token + beta * kl
    if reduction == "auto":
        reduction = "token_mean" if kind in {"dapo", "cispo"} else "sequence_mean"
    if reduction == "token_mean":
        loss = masked_mean(per_token, mask)
    elif reduction == "fixed_length":
        if not max_length:
            raise ValueError("fixed_length reduction requires max_length")
        loss = (per_token * mask).sum() / (logp.shape[0] * max_length)
    else:
        lengths = mask.sum(-1)
        active = lengths > 0
        seq_loss = (per_token * mask).sum(-1) / lengths.clamp_min(1)
        loss = (seq_loss * active).sum() / active.sum().clamp_min(1)
    outside = (ratio < 1 - clip_low) | (ratio > 1 + clip_high)
    # Clipped surrogate activity also depends on advantage sign, not only ratio bounds.
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high) * advantage
    surrogate_clipped = torch.minimum(unclipped, clipped) != unclipped
    stats = {
        "policy/ratio": masked_mean(ratio.expand_as(mask), mask).detach(),
        # Historical name: fraction of ratios outside the clip interval.
        "policy/clip_fraction": masked_mean(outside.float().expand_as(mask), mask).detach(),
        "policy/ratio_outside_fraction": masked_mean(
            outside.float().expand_as(mask), mask
        ).detach(),
        "policy/surrogate_clipped_fraction": masked_mean(
            surrogate_clipped.float().expand_as(mask), mask
        ).detach(),
        "policy/kl": masked_mean(kl, mask).detach(),
    }
    return loss, stats


def dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected).detach())
    return -F.logsigmoid(margin).mean(), margin.detach()


def opd_loss(logp, old_logp, teacher_logp, mask):
    """Sampled reverse-KL score-function estimator on student-generated tokens.

    With teacher fixed, ∇KL = E[(log pi - log teacher) ∇log pi]. The +1 term
    has zero expectation. Snapshot both gap and sampling denominator before updates.
    """
    gap = old_logp.detach() - teacher_logp.detach()
    loss = masked_mean((logp - old_logp.detach()).exp() * gap, mask)
    return loss, masked_mean(gap, mask).detach()


def gae(rewards, values, mask, gamma=1.0, lam=0.95, bootstrap=None):
    """GAE with terminal padding; bootstrap supplies V(s_T) for genuine truncations."""
    adv = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    next_value = torch.zeros_like(carry) if bootstrap is None else bootstrap
    for t in reversed(range(rewards.shape[1])):
        live = mask[:, t]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        carry = (delta + gamma * lam * carry) * live
        adv[:, t] = carry
        next_value = torch.where(live.bool(), values[:, t], next_value)
    return adv.detach(), (adv + values).detach()


def ppo_loss(
    logp, old_logp, values, old_values, advantages, returns, mask, clip=0.2, value_clip=0.2, value_coef=0.5
):
    pg, stats = policy_loss(
        logp,
        old_logp,
        advantages,
        mask,
        kind="clipped_pg",
        clip_low=clip,
        clip_high=clip,
        reduction="token_mean",
    )
    clipped = old_values.detach() + (values - old_values.detach()).clamp(-value_clip, value_clip)
    vl = 0.5 * masked_mean(
        torch.maximum((values - returns.detach()).square(), (clipped - returns.detach()).square()), mask
    )
    stats["loss/value"] = vl.detach()
    return pg + value_coef * vl, stats


def overlong_penalty(lengths, soft_limit, hard_limit):
    if not 0 <= soft_limit < hard_limit:
        raise ValueError("Require 0 <= soft limit < hard limit")
    return -((lengths - soft_limit) / (hard_limit - soft_limit)).clamp(0, 1)
