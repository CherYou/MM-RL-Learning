"""Loss extensions registered in verl; normalization is global across DP workers.

An actor processes a microbatch at a time. DDP/FSDP averages gradients, hence
each local numerator is multiplied by world_size/global_denominator. This also
handles uneven response lengths and zero-weight dispatch padding correctly.
"""

import torch
from verl.trainer.ppo.core_algos import register_policy_loss


def reduce_tokens(values, mask, config, *, kind=None):
    kind = kind or config.get("reduction", "sequence")
    if kind == "token":
        numerator = (values * mask).sum()
        denominator = config["global_tokens"]
    elif kind == "fixed":
        numerator = (values * mask).sum()
        denominator = config["global_sequences"] * config["max_new_tokens"]
    else:
        lengths = mask.sum(-1)
        numerator = ((values * mask).sum(-1) / lengths.clamp_min(1)).sum()
        denominator = config["global_sequences"]
    return numerator * config.get("world_size", 1) / max(1.0, float(denominator))


def objective(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode=None,
    config=None,
    rollout_is_weights=None,
    rollout_log_probs=None,
    *,
    kind,
):
    config = config or {}
    mask = response_mask.to(log_prob.dtype)
    delta = log_prob - old_log_prob.detach()
    if kind == "gspo":
        delta = (delta * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    ratio = delta.exp()
    low, high = config.get("clip_low", 0.2), config.get("clip_high", 0.2)
    advantage = advantages.detach()
    if kind == "opd":
        surrogate = ratio * advantage
    elif kind == "cispo":
        surrogate = ratio.clamp(1 - low, 1 + high).detach() * advantage * log_prob
    else:
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - low, 1 + high) * advantage)
    if rollout_is_weights is not None:
        surrogate = surrogate * rollout_is_weights.detach()
    loss = reduce_tokens(-surrogate, mask, config)
    weight = mask.sum().clamp_min(1)
    return loss, {
        "policy/ratio": float((ratio.expand_as(mask).detach() * mask).sum() / weight),
        "policy/clip_fraction": float(
            (((ratio < 1 - low) | (ratio > 1 + high)).expand_as(mask) * mask).sum() / weight
        ),
    }


def _register(kind):
    @register_policy_loss("lab_" + kind)
    def loss_fn(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        loss_agg_mode=None,
        config=None,
        rollout_is_weights=None,
        rollout_log_probs=None,
    ):
        return objective(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            loss_agg_mode,
            config,
            rollout_is_weights,
            rollout_log_probs,
            kind=kind,
        )

    return loss_fn


REGISTERED = {kind: _register(kind) for kind in ("grpo", "gspo", "dapo", "opd", "cispo", "dr_grpo")}


def add_reference_kl(loss, log_prob, reference, mask, config):
    if not config.get("beta", 0):
        return loss, 0.0
    if reference is None:
        raise ValueError("A frozen reference is required for KL regularization")
    gap = reference.detach() - log_prob
    kl = gap.exp() - gap - 1
    term = reduce_tokens(kl, mask, config)
    return loss + config["beta"] * term, float(term.detach())
