"""TEMPO state replay with explicit behavior-prefix probabilities and correction."""

import copy
import math
import random
import torch

from agentic_rl.models import Sample
from agentic_rl.tempo import parse_value, signals, critic_rollouts
from .agent_loop import collect


class ReplayStore:
    def __init__(self, seed=42, capacity=128):
        self.states = []
        self.capacity = capacity
        self.rng = random.Random(seed)

    def pop(self):
        return self.states.pop(self.rng.randrange(len(self.states))) if self.states else None

    def add(self, row, sample, parent=None):
        if sample.metadata.get("stop_reason") != "macro_boundary":
            return
        mask = list(sample.mask)
        behavior = list(sample.old_logp)
        if parent:
            n = len(parent["tokens"])
            mask[:n] = parent["behavior_mask"]
            behavior[: n - 1] = parent["behavior_logp"]
        self.states.append(
            {
                "row": copy.deepcopy(row),
                "tokens": list(sample.tokens),
                "actions": list(sample.metadata["actions"]),
                "observations": list(sample.metadata["observations"]),
                "behavior_mask": mask,
                "behavior_logp": behavior,
                "depth": (parent["depth"] if parent else 0) + 1,
            }
        )
        self.states = self.states[-self.capacity :]

    def state_dict(self):
        return {"states": self.states, "capacity": self.capacity, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.states, self.capacity = state["states"], state["capacity"]
        self.rng.setstate(state["rng"])


def prefix_correction(policy, state, config):
    if state is None:
        return 1.0, 0.0
    mask = state["behavior_mask"]
    # Restore the original sampling mask only for rescoring, never for the current update.
    first = next((i for i, x in enumerate(mask) if x), len(mask))
    sample = Sample(state["tokens"], mask, first)
    current, _, _ = policy.score([sample])
    delta = current[0, : len(sample.tokens) - 1] - torch.tensor(state["behavior_logp"])
    log_ratio = float((delta * torch.tensor(mask[1:])).sum())
    bound = config.get("prefix_is_clip")
    if bound is not None:
        if bound < 1:
            raise ValueError("prefix_is_clip must be >=1, or null for exact importance sampling")
        corrected = max(-math.log(bound), min(math.log(bound), log_ratio))
    else:
        corrected = log_ratio
    weight = math.exp(corrected)
    if not math.isfinite(weight):
        raise FloatingPointError("TEMPO prefix weight overflow; use explicit bounded correction")
    return weight, log_ratio


def tempo_batch(policy, row, config, store, step):
    warm = step < config.get("warmup_steps", 1)
    state = store.pop() if not warm else None
    if state:
        row = state["row"]
    prefix_weight, prefix_log_ratio = prefix_correction(policy, state, config)
    horizon = config["max_turns"] if warm else config.get("macro_horizon", 2)
    if state:
        horizon = min(horizon, config["max_turns"] - len(state["actions"]))
        if horizon <= 0:
            raise ValueError("Replay store contains an exhausted episode")
    branches, _ = collect(
        policy, [row] * config["group_size"], config, step=step, initial=state, turn_limit=horizon
    )
    endpoint_values = []
    failures = 0
    for branch in branches:
        if len(branch.metadata["actions"]) >= config["max_turns"] and not branch.metadata.get("done"):
            branch.metadata["stop_reason"] = "turn_limit"
        if warm or branch.metadata.get("stop_reason") != "macro_boundary":
            endpoint_values.append(0.0)
        else:
            parsed = [parse_value(s.text) for s in critic_rollouts(policy, {"tokens": branch.tokens}, config)]
            good = [v for v in parsed if v is not None]
            failures += len(parsed) - len(good)
            endpoint_values.append(sum(good) / len(good) if good else 0.5)
    returns, advantages, target = signals(
        [s.reward for s in branches],
        endpoint_values,
        [warm or s.metadata.get("stop_reason") != "macro_boundary" for s in branches],
    )
    for branch, a, value in zip(branches, advantages, returns, strict=True):
        branch.advantages = [float(a) * m for m in branch.mask]
        branch.metadata.update(
            kind="actor",
            return_value=float(value),
            prefix_is_weight=prefix_weight,
            prefix_log_ratio=prefix_log_ratio,
            replayed=state is not None,
        )
        if not warm:
            store.add(row, branch, state)
    start = {"tokens": branches[0].tokens[: branches[0].prompt_length]}
    critic = critic_rollouts(policy, start, config)
    scores = [-1.0 if (v := parse_value(s.text)) is None else -abs(v - float(target)) for s in critic]
    mean = sum(scores) / len(scores)
    for sample, value in zip(critic, scores, strict=True):
        sample.reward = value
        sample.advantages = [(value - mean) * m for m in sample.mask]
        sample.metadata["prefix_is_weight"] = prefix_weight
    return branches + critic, {
        "tempo/target": float(target),
        "tempo/endpoint_value": sum(endpoint_values) / len(endpoint_values),
        "tempo/critic_parse_failures": failures + sum(parse_value(s.text) is None for s in critic),
        "tempo/state_store_size": len(store.states),
        "tempo/warmup": float(warm),
        "tempo/replayed": float(state is not None),
        "tempo/prefix_is_weight": prefix_weight,
        "tempo/prefix_log_ratio": prefix_log_ratio,
    }
