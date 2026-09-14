"""Harness-RL: interface records, session prefix trees and activation-based CAPO.

Local structural choice: route MLP expansion/projection units; shared units receive
the sum of both gradients. Freeze parameters outside these units. The algorithm
does not flatten unrelated sessions or train on frozen workers' outputs.
"""

from dataclasses import dataclass
import json
import math
import re
import torch

from .data import read_jsonl
from .models import Sample
from .environments import LocalSearch, python_tool
from .rewards import token_f1


@dataclass
class CallRecord:
    rollout_id: str
    session_id: str
    role: str
    call_id: str
    parent_call_id: str | None
    sample: Sample
    action_mask: list[float]
    args_mask: list[float]
    process_reward: float = 0.0

    def as_dict(self):
        return {
            "rollout_id": self.rollout_id,
            "session_id": self.session_id,
            "role": self.role,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "token_in": self.sample.tokens[: self.sample.prompt_length],
            "token_out": self.sample.tokens[self.sample.prompt_length :],
            "rollout_logprobs": self.sample.old_logp[self.sample.prompt_length - 1 :],
            "action_mask": self.action_mask,
            "args_mask": self.args_mask,
            "process_reward": self.process_reward,
            "reward": self.sample.reward,
        }


def prefix_trees(records):
    """One token trie per (rollout, session), output spans remain separate records.

    Repeated calls with identical text remain distinct learning samples. The trie
    shares only context nodes, never discards a captured call or merges its reward.
    """
    trees = {}
    for r in records:
        key = (r.rollout_id, r.session_id)
        tree = trees.setdefault(key, {"nodes": [{"children": {}, "outputs": []}]})
        node = 0
        for token in r.sample.tokens:
            children = tree["nodes"][node]["children"]
            if token not in children:
                children[token] = len(tree["nodes"])
                tree["nodes"].append({"children": {}, "outputs": []})
            node = children[token]
        tree["nodes"][node]["outputs"].append(r.call_id)
    return trees


def structured_masks(tokenizer, output_ids):
    """Identify JSON action/args spans on exact generated IDs, without re-tokenizing.

    Prefix decoding supplies character offsets. Boundary tokens may straddle a
    quote and a value; classify by overlap, with action taking precedence.
    """
    text = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    action = re.search(r'"action"\s*:\s*"([^"\\]*)"', text)
    args = re.search(r'"args"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    am = [0.0] * len(output_ids)
    gm = [0.0] * len(output_ids)
    # Fast tokenizer decode may replace an incomplete byte prefix. Enforce monotone offsets.
    ends = [
        len(
            tokenizer.decode(
                output_ids[: i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )
        for i in range(len(output_ids))
    ]
    start = 0
    for i, end in enumerate(ends):
        end = max(start, end)
        if action and start < action.end(1) and end > action.start(1):
            am[i] = 1.0
        elif args and start < args.end(1) and end > args.start(1):
            gm[i] = 1.0
        start = end
    return am, gm


def capture(policy, ids, out, *, rollout_id, session_id, call_id, role="central", parent=None):
    action, args = structured_masks(policy.tokenizer, out)
    sample = Sample(
        ids + out,
        [0.0] * len(ids) + [1.0] * len(out),
        len(ids),
        text=policy.tokenizer.decode(out, skip_special_tokens=True),
        turns=[(len(ids), len(ids) + len(out))],
    )
    policy.snapshot(sample)
    return CallRecord(
        rollout_id,
        session_id,
        role,
        call_id,
        parent,
        sample,
        [0.0] * len(ids) + action,
        [0.0] * len(ids) + args,
    )


def harness_rollout(policy, row, config, rollout_id):
    """Central agent dispatches to deterministic frozen retrieval/calculation workers."""
    system = (
        'Return JSON {"action":"Reason|Search|Calculate|Summary","args":"text"}. '
        "Search dispatches to a retrieval worker. Calculate dispatches numerical Python. "
        "Summary ends the task and its args must contain the final answer."
    )
    ids, _ = policy.prompt(row["prompt"], system)
    search = LocalSearch(config.get("corpus", "data/search/corpus.jsonl"))
    records = []
    final = ""
    artifact = False
    for step in range(config["max_turns"]):
        if len(ids) + config["max_new_tokens"] > config.get("max_context_tokens", 4096):
            break
        out = policy.generate(ids, config["max_new_tokens"])
        record = capture(policy, ids, out, rollout_id=rollout_id, session_id="central", call_id=f"c{step}")
        records.append(record)
        try:
            decoded = json.loads(record.sample.text)
            action, args = decoded["action"], decoded["args"]
            if not isinstance(args, str):
                raise ValueError("args must be a string")
        except (ValueError, KeyError, TypeError):
            action, args = "Invalid", ""
        if action == "Summary":
            final = args
            artifact = True
            break
        if action == "Search":
            observation = search.search(args)
        elif action == "Calculate":
            observation = python_tool(args)
        elif action == "Reason":
            observation = "Continue planning or dispatch a worker."
        else:
            observation = "Invalid JSON decision. Return action and args."
        record.process_reward = float(
            action in {"Search", "Calculate"} and not observation.startswith(("ToolError", "No documents"))
        )
        # Workers have separate sessions and are frozen. Their outputs are observations,
        # not model samples with invented sampling logprobs.
        record.sample.metadata["worker_call"] = {
            "session_id": f"worker-{step}",
            "parent_call_id": record.call_id,
            "role": "frozen_tool",
            "input": args,
            "output": observation,
        }
        ids = ids + out + policy.encode("\nWorker observation: " + observation[:2000] + "\nAssistant: ")
    score = float(artifact) * max(token_f1(final, str(x)) for x in row.get("answers", [row["answer"]]))
    for record in records:
        record.sample.reward = score
    return records, score


def assign_advantages(groups, process_coef=0.1):
    """Normalize outcomes by query group and process rewards by aligned decision index."""
    from .losses import group_advantages

    outcome = torch.tensor([g[0].sample.reward if g else 0.0 for g in groups])
    adv = group_advantages(outcome, len(groups))
    for step in range(max(map(len, groups), default=0)):
        eligible = [(i, g[step]) for i, g in enumerate(groups) if step < len(g)]
        rewards = torch.tensor([r.process_reward for _, r in eligible])
        proc = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-4)
        for (i, r), pa in zip(eligible, proc):
            a = float(adv[i] + process_coef * pa)
            r.sample.advantages = [a * (x + y) for x, y in zip(r.action_mask, r.args_mask)]


class CAPORouter:
    """Fixed Top-K MLP unit partitions from successful probing trajectories."""

    def __init__(self, policy, fraction=0.25):
        if not 0 < fraction <= 1:
            raise ValueError("CAPO fraction must be in (0,1]")
        self.policy = policy
        self.fraction = fraction
        self.masks = {}

    @torch.no_grad()
    def probe(self, records):
        successful = [r for r in records if r.sample.reward > 0 and any(r.action_mask) and any(r.args_mask)]
        if not successful:
            raise ValueError(
                "CAPO needs successful structured probes; provide --probe-data or collect successful rollouts"
            )
        units = {n: m for n, m in self.policy.lm.named_modules() if n.endswith((".c_fc", ".up_proj"))}
        stats = {n: [None, None] for n in units}
        for r in successful:
            handles = []
            for name, module in units.items():

                def hook(_module, _input, output, name=name):
                    positive = output[0, :-1].float().clamp_min(0)
                    for index, mask in enumerate((r.action_mask, r.args_mask)):
                        weight = torch.tensor(mask[1:], device=positive.device)[:, None]
                        importance = (positive * weight).sum(0) / weight.sum().clamp_min(1)
                        stats[name][index] = (
                            importance if stats[name][index] is None else stats[name][index] + importance
                        )

                handles.append(module.register_forward_hook(hook))
            try:
                self.policy.score([r.sample])
            finally:
                for handle in handles:
                    handle.remove()
        parameters = dict(self.policy.lm.named_parameters())
        for name, (action, args) in stats.items():
            if action is None:
                continue
            n = action.numel()
            k = max(1, math.ceil(n * self.fraction))
            ua = torch.zeros(n, device=action.device)
            ug = ua.clone()
            ua[action.topk(k).indices] = 1
            ug[args.topk(k).indices] = 1
            for suffix in ("weight", "bias"):
                key = name + "." + suffix
                if key in parameters:
                    p = parameters[key]
                    if p.ndim == 1:
                        ma, mg = ua, ug
                    elif name.endswith(".c_fc"):
                        ma, mg = ua[None].expand_as(p), ug[None].expand_as(p)
                    else:
                        ma, mg = ua[:, None].expand_as(p), ug[:, None].expand_as(p)
                    self.masks[key] = (ma, mg)
            projection = name.rsplit(".", 1)[0] + (
                ".c_proj.weight" if name.endswith(".c_fc") else ".down_proj.weight"
            )
            if projection in parameters:
                p = parameters[projection]
                self.masks[projection] = (
                    (ua[:, None].expand_as(p), ug[:, None].expand_as(p))
                    if name.endswith(".c_fc")
                    else (ua[None].expand_as(p), ug[None].expand_as(p))
                )
        if not self.masks:
            raise ValueError("No supported MLP units found; use GPT-2/Llama/Qwen without LoRA for CAPO")

    def backward(self, action_loss, args_loss):
        named = list(self.policy.lm.named_parameters())
        params = [p for _, p in named]
        ga = torch.autograd.grad(action_loss, params, retain_graph=True, allow_unused=True)
        gg = torch.autograd.grad(args_loss, params, allow_unused=True)
        dot = norma = normg = 0.0
        for (name, p), a, g in zip(named, ga, gg):
            p.grad = None
            if name not in self.masks:
                continue
            ma, mg = self.masks[name]
            if a is not None and g is not None:
                dot += float((a * g).sum())
                norma += float(a.square().sum())
                normg += float(g.square().sum())
                p.grad = ma * a + mg * g
            elif a is not None:
                p.grad = ma * a
            elif g is not None:
                p.grad = mg * g
        return {
            "capo/gradient_cosine": dot / max(1e-12, math.sqrt(norma * normg)),
            "capo/routed_parameters": sum(int(((a + g) > 0).sum()) for a, g in self.masks.values()),
        }

    def save(self, path):
        torch.save({k: (a.cpu(), g.cpu()) for k, (a, g) in self.masks.items()}, path)


def load_probes(policy, path):
    probes = []
    for i, row in enumerate(read_jsonl(path)):
        if "token_in" in row:
            ids, out = row["token_in"], row["token_out"]
        else:
            ids, _ = policy.prompt(row["prompt"])
            out = policy.encode(row["response"])
        r = capture(policy, ids, out, rollout_id=f"probe-{i}", session_id="central", call_id="0")
        r.sample.reward = float(row["reward"])
        probes.append(r)
    return probes
