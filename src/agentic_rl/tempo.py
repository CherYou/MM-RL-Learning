"""TEMPO macro-step optimization with the SAME generative actor/critic policy.

Implementation follows the upstream algorithm-level recipe. A macro boundary is
bootstrapped; a real terminal state has zero future value. Replay verifies state.
"""

import random
import re
import torch
from .models import Sample
from .rollout import agent_rollout


def parse_value(text):
    matches = re.findall(r"<value>\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*</value>", text)
    return float(matches[0]) if len(matches) == 1 else None


def signals(rewards, values, terminals):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    v = torch.as_tensor(values, dtype=torch.float32)
    returns = r + v * (1 - torch.as_tensor(terminals, dtype=torch.float32))
    target = returns.mean()
    return returns, returns - target, target


def critic_rollouts(policy, state, config):
    prompt = (
        "Estimate probability of eventual task success from this household state. "
        "Reason briefly and output exactly one <value>number in [0,1]</value>.\n"
        + policy.tokenizer.decode(state["tokens"], skip_special_tokens=True)
    )
    if state.get("walkthrough"):
        prompt += "\nPrivileged training-only walkthrough: " + state["walkthrough"]
    ids, _ = policy.prompt(prompt)
    samples = []
    for _ in range(config.get("critic_samples", 4)):
        out = policy.generate(ids, config.get("critic_tokens", config["max_new_tokens"]))
        s = Sample(
            ids + out,
            [0.0] * len(ids) + [1.0] * len(out),
            len(ids),
            text=policy.tokenizer.decode(out, skip_special_tokens=True),
            metadata={"kind": "critic"},
        )
        samples.append(policy.snapshot(s))
    return samples


class StateStore:
    def __init__(self, capacity=128, seed=42):
        self.states = []
        self.capacity = capacity
        self.rng = random.Random(seed)

    def add(self, row, sample):
        if sample.metadata.get("stop_reason") != "macro_boundary":
            return
        state = {
            "row": row,
            "tokens": list(sample.tokens),
            "actions": sample.metadata["actions"],
            "observations": sample.metadata["observations"],
        }
        # End of segment already contains the last observation and Assistant prefix.
        self.states.append(state)
        self.states = self.states[-self.capacity :]

    def pop(self):
        return self.states.pop(self.rng.randrange(len(self.states))) if self.states else None


def tempo_batch(policy, row, config, store, step):
    warm = step < config.get("warmup_steps", 1)
    state = store.pop() if not warm else None
    if state:
        row = state["row"]
    horizon = config["max_turns"] if warm else config.get("macro_horizon", 2)
    branches = [
        agent_rollout(policy, row, config, initial=state, turn_limit=horizon)
        for _ in range(config["group_size"])
    ]
    endpoint_values = []
    failures = 0
    for branch in branches:
        if branch.metadata.get("stop_reason") != "macro_boundary" or warm:
            endpoint_values.append(0.0)
        else:
            estimates = critic_rollouts(policy, {"tokens": branch.tokens}, config)
            parsed = [parse_value(s.text) for s in estimates]
            good = [v for v in parsed if v is not None]
            failures += len(parsed) - len(good)
            endpoint_values.append(sum(good) / len(good) if good else 0.5)
    returns, advantages, target = signals(
        [s.reward for s in branches],
        endpoint_values,
        [s.metadata.get("stop_reason") != "macro_boundary" or warm for s in branches],
    )
    for branch, a, return_value in zip(branches, advantages, returns):
        branch.advantages = [float(a) * m for m in branch.mask]
        branch.metadata.update(kind="actor", return_value=float(return_value))
        if not warm:
            store.add(row, branch)
    start = {"tokens": branches[0].tokens[: branches[0].prompt_length]}
    critic = critic_rollouts(policy, start, config)
    scores = []
    for s in critic:
        v = parse_value(s.text)
        scores.append(-1.0 if v is None else -abs(v - float(target)))
    mean = sum(scores) / len(scores)
    for s, r in zip(critic, scores):
        s.reward = r
        s.advantages = [(r - mean) * m for m in s.mask]
    return branches + critic, {
        "tempo/target": target,
        "tempo/endpoint_value": sum(endpoint_values) / len(endpoint_values),
        "tempo/critic_parse_failures": failures + sum(parse_value(s.text) is None for s in critic),
        "tempo/state_store_size": len(store.states),
        "tempo/warmup": float(warm),
    }
