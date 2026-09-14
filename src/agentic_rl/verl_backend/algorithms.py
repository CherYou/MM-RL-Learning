"""Algorithm scheduling and credit assignment; all model calls go to verl workers."""

import torch

from agentic_rl.data import read_jsonl
from agentic_rl.losses import group_advantages, overlong_penalty, gae, masked_mean
from agentic_rl.trainers import teacher_logprobs
from agentic_rl.agentopsd import reshape_turn_advantages
from agentic_rl.harness import assign_advantages, prefix_trees
from .agent_loop import collect
from .protocol import pack
from .tempo import ReplayStore, tempo_batch


class AlgorithmBatchBuilder:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.policy = runtime.policy
        self.rows = read_jsonl(config["dataset"])
        self.eval_rows = read_jsonl(config["eval_dataset"])
        if {r["prompt"] for r in self.rows} & {r["prompt"] for r in self.eval_rows}:
            raise ValueError("Training and evaluation prompts overlap")
        self.general = (
            read_jsonl(config["general_dataset"]) if config["algorithm"] in {"sar-opd", "idt-opd"} else None
        )
        if self.general and {r["prompt"] for r in self.general} & {r["prompt"] for r in self.eval_rows}:
            raise ValueError("General recovery data overlaps evaluation")
        self.teacher = (
            runtime.remote_policy("teacher")
            if config["algorithm"] in {"opd", "sar-opd", "idt-opd", "opsd"}
            else None
        )
        self.reference = (
            runtime.remote_policy("reference")
            if config.get("beta", 0) > 0 or config["algorithm"] in {"ppo", "sar-opd", "idt-opd"}
            else None
        )
        self.store = ReplayStore(config["seed"], config.get("state_store_capacity", 128))

    def build(self, step):
        cfg, policy = self.config, self.policy
        algo, group_size = cfg["algorithm"], cfg["group_size"]
        general_phase = (algo == "idt-opd" and step % 2 == 1) or (
            algo == "sar-opd" and step >= cfg.get("medical_steps", max(1, cfg["steps"] // 2))
        )
        pool = self.general if general_phase else self.rows
        selected = [pool[(step * cfg["batch_size"] + i) % len(pool)] for i in range(cfg["batch_size"])]
        metrics, artifacts = {}, {}
        if algo in {"sar-opd", "idt-opd"}:
            metrics["distill/general_phase"] = float(general_phase)
        samples, batch_rows = [], []
        if algo == "tempo":
            for row in selected:
                branches, stats = tempo_batch(policy, row, cfg, self.store, step)
                samples.extend(branches)
                batch_rows.extend([row] * len(branches))
                metrics.update(stats)
        elif algo == "harness-rl":
            _, groups = collect(policy, [row for row in selected for _ in range(group_size)], cfg, step=step)
            for offset in range(0, len(groups), group_size):
                assign_advantages(groups[offset : offset + group_size], cfg.get("process_coef", 0.1))
            records = [r for group in groups for r in group]
            samples = [r.sample for r in records]
            artifacts["interface"] = [r.as_dict() for r in records]
            artifacts["trees"] = {str(k): v for k, v in prefix_trees(records).items()}
        else:
            active = selected
            attempts, degenerate = 0, 0
            while active and attempts < cfg.get("max_resample_batches", 8):
                attempts += 1
                expanded = [row for row in active for _ in range(group_size)]
                generated, _ = collect(policy, expanded, cfg, step=step)
                retry = []
                for i, row in enumerate(active):
                    group = generated[i * group_size : (i + 1) * group_size]
                    if algo == "dapo":
                        if max(s.reward for s in group) == min(s.reward for s in group):
                            degenerate += 1
                            retry.append(pool[(step + attempts + degenerate) % len(pool)])
                            continue
                        penalties = overlong_penalty(
                            torch.tensor([sum(s.mask) for s in group]),
                            cfg["soft_length"],
                            cfg["max_new_tokens"],
                        )
                        for s, p in zip(group, penalties, strict=True):
                            s.metadata["unshaped_reward"] = s.reward
                            s.reward += float(p)
                    samples.extend(group)
                    batch_rows.extend([row] * len(group))
                active = retry if algo == "dapo" else []
            metrics.update({"sampling/attempts": attempts, "sampling/degenerate_groups": degenerate})
        if not samples:
            return None, [], metrics, artifacts
        if self.runtime.generator is not None:
            self.runtime.generator.ensure_train()
        data = pack(samples, policy.tokenizer.pad_token_id)
        batch = data.batch
        mask, old = batch["response_mask"], batch["old_log_probs"]
        rewards = torch.tensor([s.reward for s in samples])
        metrics["reward/mean"] = float(rewards.mean())
        if self.reference:
            batch["ref_log_prob"] = self.reference.score(samples)[0]
        kind = "gspo" if algo == "gspo" else "dapo" if algo in {"dapo", "ppo", "harness-rl"} else "grpo"
        if algo in {"opd", "sar-opd", "idt-opd", "opsd"}:
            teacher = self.reference if general_phase else self.teacher
            tlp = teacher_logprobs(
                teacher, policy, samples, batch_rows, "solution" if algo == "opsd" else None
            )
            batch["teacher_log_probs"] = tlp
            batch["advantages"] = (tlp - old).detach()
            metrics["distill/sampled_reverse_kl"] = float(masked_mean(old - tlp, mask))
            kind = "opd"
        elif algo == "ppo":
            _, _, values = policy.score(samples, values=True)
            token_rewards = -cfg.get("beta", 0.02) * (old - batch["ref_log_prob"]) * mask
            for i, s in enumerate(samples):
                last = max(j for j, m in enumerate(s.mask[1:]) if m)
                token_rewards[i, last] += s.reward
            advantages, returns = gae(token_rewards, values, mask, lam=cfg.get("gae_lambda", 0.95))
            mean = masked_mean(advantages, mask)
            std = masked_mean((advantages - mean).square(), mask).sqrt()
            batch["advantages"] = (advantages - mean) / (std + 1e-8) * mask
            batch["old_values"], batch["returns"] = values, returns
        elif algo == "harness-rl":
            batch["action_mask"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r.action_mask[1:]) for r in records], batch_first=True
            )
            batch["args_mask"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r.args_mask[1:]) for r in records], batch_first=True
            )
        elif algo != "tempo":
            advantages = group_advantages(rewards, group_size, scale=cfg.get("scale_rewards", True))
            batch["advantages"] = advantages[:, None].expand_as(old).clone()
            if algo == "agentopsd":
                tlp = teacher_logprobs(policy, policy, samples, batch_rows, "skill")
                turn_adv = torch.zeros_like(old)
                for i, (sample, a) in enumerate(zip(samples, advantages, strict=True)):
                    evidence = [
                        float((tlp[i, start - 1 : end - 1] - old[i, start - 1 : end - 1]).sum())
                        for start, end in sample.turns
                    ]
                    begin = i // group_size * group_size
                    credit = reshape_turn_advantages(
                        float(a), float(rewards[begin : begin + group_size].mean()), evidence
                    )
                    for (start, end), turn in zip(sample.turns, credit.turns, strict=True):
                        turn_adv[i, start - 1 : end - 1] = turn.advantage
                    sample.metadata.update(
                        turn_evidence=evidence, turn_advantages=[x.advantage for x in credit.turns]
                    )
                batch["advantages"] = turn_adv
        if cfg.get("mask_truncated", False):
            for i, sample in enumerate(samples):
                if sample.metadata.get("truncated"):
                    batch["response_mask"][i].zero_()
        mask = batch["response_mask"]
        reduction = "token" if kind in {"dapo", "opd"} else "sequence"
        loss_config = {
            "world_size": self.runtime.world_size,
            "global_tokens": float(mask.sum()),
            "global_sequences": int((mask.sum(-1) > 0).sum()),
            "reduction": reduction,
        }
        if algo == "harness-rl":
            loss_config.update(
                action_tokens=float(batch["action_mask"].sum()), args_tokens=float(batch["args_mask"].sum())
            )
        if algo == "tempo":
            batch["prefix_is_weights"] = torch.tensor(
                [s.metadata.get("prefix_is_weight", 1.0) for s in samples]
            )[:, None].expand_as(old)
        data.meta_info.update(loss_kind=kind, loss_config=loss_config, step=step)
        return data, samples, metrics, artifacts
