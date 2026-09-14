"""Readable PyTorch/Accelerate trainers for objectives and agent environments.

TRL baselines are in trl_backend.py. This module makes the mechanisms not exposed
by a standard trainer explicit: distillation, dynamic sampling, tool masks, TEMPO
and CAPO. Single-process by design; no remote service or GPU reservation occurs.
"""

import json
import random
import time
import torch
from accelerate import Accelerator

from .data import ROOT, read_jsonl
from .models import Policy, VisionPolicy, Sample
from .logging import RunLogger
from .losses import (
    group_advantages,
    masked_mean,
    policy_loss,
    opd_loss,
    dpo_loss,
    gae,
    ppo_loss,
    overlong_penalty,
)
from .rollout import single_rollout, agent_rollout, MATH_SYSTEM
from .agentopsd import reshape_turn_advantages
from .tempo import StateStore, tempo_batch
from .harness import CAPORouter, load_probes, harness_rollout, assign_advantages, prefix_trees


def padded_old(samples, device):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s.old_logp, device=device) for s in samples], batch_first=True
    )


def supervised_sample(policy, row, key, max_tokens):
    ids, _ = policy.prompt(row["prompt"], MATH_SYSTEM)
    output = policy.encode(row[key])[:max_tokens] + [policy.tokenizer.eos_token_id]
    return Sample(ids + output, [0.0] * len(ids) + [1.0] * len(output), len(ids), text=row[key])


@torch.no_grad()
def teacher_logprobs(teacher, student, samples, rows, privileged=None):
    if teacher.tokenizer.get_vocab() != student.tokenizer.get_vocab():
        raise ValueError("Token-level OPD requires identical student/teacher vocabularies")
    result = []
    for s, row in zip(samples, rows, strict=True):
        if privileged == "solution":
            ids, _ = teacher.prompt(
                row["prompt"], MATH_SYSTEM + "\nReference solution (teacher only): " + row["solution"]
            )
        elif privileged == "skill":
            skill_root = ROOT / "data/skills"
            skill = (skill_root / ("general_skills.md" if row.get("task_type") else "general.md")).read_text()
            mapping = {
                "pick_and_place_simple": "pick_and_place.md",
                "look_at_obj_in_light": "look_at_obj_in_light.md",
                "pick_clean_then_place_in_recep": "clean.md",
                "pick_heat_then_place_in_recep": "heat.md",
                "pick_cool_then_place_in_recep": "cool.md",
                "pick_two_obj_and_place": "pick_and_place.md",
            }
            task_skill = skill_root / mapping.get(row.get("task_type"), "general.md")
            if task_skill.exists() and task_skill.name != "general.md":
                skill += "\n" + task_skill.read_text()
            ids = (
                teacher.encode("Teacher-only household skill:\n" + skill + "\n") + s.tokens[: s.prompt_length]
            )
        else:
            ids = s.tokens[: s.prompt_length]
        suffix = s.tokens[s.prompt_length :]
        t = Sample(ids + suffix, [0.0] * len(ids) + s.mask[s.prompt_length :], len(ids))
        lp, _, _ = teacher.score([t])
        aligned = torch.zeros(len(s.tokens) - 1, device=student.device)
        aligned[s.prompt_length - 1 :] = lp[0, len(ids) - 1 :].to(student.device)
        result.append(aligned)
    return torch.nn.utils.rnn.pad_sequence(result, batch_first=True)


def run_native(config):
    torch.set_num_threads(config.get("cpu_threads", 2))
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    accelerator = Accelerator(cpu=config["device"] == "cpu", mixed_precision="no")
    if accelerator.num_processes != 1:
        raise ValueError("Native learning backend supports one process; use TRL for distributed training")
    cls = VisionPolicy if config["algorithm"] == "vision-grpo" else Policy
    policy = cls(config["model"], config["device"], config["seed"], config.get("lora_rank", 0))
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0.0
    )
    rows = read_jsonl(config["dataset"])
    eval_rows = read_jsonl(config["eval_dataset"])
    if {r["prompt"] for r in rows} & {r["prompt"] for r in eval_rows}:
        raise ValueError("Training and evaluation prompt sets overlap")
    algo = config["algorithm"]
    reference = (
        policy.frozen_copy()
        if config.get("beta", 0) > 0 or algo in {"dpo", "ppo", "sar-opd", "idt-opd"}
        else None
    )
    teacher = None
    if algo in {"opd", "sar-opd", "idt-opd"}:
        teacher = Policy(config["teacher_model"], config["device"], config["seed"] + 1)
        teacher.eval().requires_grad_(False)
    if algo == "opsd":
        teacher = policy.frozen_copy()  # Fixed step-0 parameters, privileged context only.
    general_rows = read_jsonl(config["general_dataset"]) if algo in {"sar-opd", "idt-opd"} else None
    logger = RunLogger(ROOT / config["output"], config)
    store = StateStore(seed=config["seed"])
    router = None
    if algo == "harness-rl" and config.get("capo", True):
        if config.get("lora_rank", 0):
            raise ValueError("CAPO routes full MLP units; this implementation requires lora_rank=0")
        router = CAPORouter(policy, config.get("capo_fraction", 0.25))
        router.probe(load_probes(policy, config["probe_data"]))
        router.save(logger.path / "capo-partitions.pt")
    started = time.perf_counter()
    try:
        for step in range(config["steps"]):
            metrics = {}
            optimizer.zero_grad(set_to_none=True)
            general_phase = (algo == "idt-opd" and step % 2 == 1) or (
                algo == "sar-opd" and step >= config.get("medical_steps", max(1, config["steps"] // 2))
            )
            pool = general_rows if general_phase else rows
            selected = [
                pool[(step * config["batch_size"] + i) % len(pool)] for i in range(config["batch_size"])
            ]
            if algo in {"sar-opd", "idt-opd"}:
                metrics["distill/general_phase"] = float(general_phase)
            if algo in {"sft", "dpo"}:
                if algo == "sft":
                    samples = [
                        supervised_sample(policy, r, "solution", config["max_new_tokens"]) for r in selected
                    ]
                    lp, mask, _ = policy.score(samples)
                    loss = masked_mean(-lp, mask)
                else:
                    chosen = [
                        supervised_sample(policy, r, "chosen", config["max_new_tokens"]) for r in selected
                    ]
                    rejected = [
                        supervised_sample(policy, r, "rejected", config["max_new_tokens"]) for r in selected
                    ]
                    samples = chosen + rejected
                    lp, mask, _ = policy.score(samples)
                    with torch.no_grad():
                        rp, _, _ = reference.score(samples)
                    sums = (lp * mask).sum(-1)
                    refs = (rp * mask).sum(-1)
                    n = len(chosen)
                    loss, margin = dpo_loss(
                        sums[:n], sums[n:], refs[:n], refs[n:], config.get("dpo_beta", 0.1)
                    )
                    metrics["dpo/margin"] = margin.mean()
                accelerator.backward(loss)
                metrics["loss/total"] = loss.detach()
            elif algo == "harness-rl":
                all_records = []
                for i, row in enumerate(selected):
                    groups = [
                        harness_rollout(policy, row, config, f"{step}-{i}-{g}")[0]
                        for g in range(config["group_size"])
                    ]
                    assign_advantages(groups, config.get("process_coef", 0.1))
                    all_records.extend(r for group in groups for r in group)
                samples = [r.sample for r in all_records]
                if not samples:
                    raise ValueError("No interface calls captured")
                trees = prefix_trees(all_records)
                (logger.path / f"interface-{step:04d}.json").write_text(
                    json.dumps([r.as_dict() for r in all_records], ensure_ascii=False, indent=2)
                )
                (logger.path / f"trees-{step:04d}.json").write_text(
                    json.dumps({str(k): v for k, v in trees.items()})
                )
                lp, mask, _ = policy.score(samples)
                old = padded_old(samples, policy.device)
                adv = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(s.advantages[1:], device=policy.device) for s in samples], batch_first=True
                )
                am = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(r.action_mask[1:], device=policy.device) for r in all_records],
                    batch_first=True,
                )
                gm = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(r.args_mask[1:], device=policy.device) for r in all_records],
                    batch_first=True,
                )
                la, _ = policy_loss(lp, old, adv, am, kind="dapo")
                lg, _ = policy_loss(lp, old, adv, gm, kind="dapo")
                if router:
                    metrics.update(router.backward(la, lg))
                else:
                    accelerator.backward(la + lg)
                metrics.update(
                    {
                        "loss/total": (la + lg).detach(),
                        "harness/action_tokens": am.sum(),
                        "harness/args_tokens": gm.sum(),
                        "reward/mean": sum(s.reward for s in samples) / len(samples),
                    }
                )
            else:
                batch_rows = []
                samples = []
                if algo == "tempo":
                    for row in selected:
                        batch, stats = tempo_batch(policy, row, config, store, step)
                        samples += batch
                        batch_rows += [row] * len(batch)
                        metrics.update(stats)
                else:
                    sampler = (
                        agent_rollout
                        if algo in {"search-r1", "retool", "alfworld", "agentopsd"}
                        else single_rollout
                    )
                    active = selected
                    attempts = 0
                    degenerate = 0
                    while active and attempts < config.get("max_resample_batches", 8):
                        attempts += 1
                        retry = []
                        for row in active:
                            group = [sampler(policy, row, config) for _ in range(config["group_size"])]
                            if algo == "dapo":
                                # Dynamic sampling uses task reward BEFORE length shaping.
                                if max(s.reward for s in group) == min(s.reward for s in group):
                                    degenerate += 1
                                    retry.append(pool[(step + attempts + degenerate) % len(pool)])
                                    continue
                                lengths = torch.tensor([sum(s.mask) for s in group])
                                penalties = overlong_penalty(
                                    lengths,
                                    config.get("soft_length", int(config["max_new_tokens"] * 0.75)),
                                    config["max_new_tokens"],
                                )
                                for s, penalty in zip(group, penalties):
                                    s.reward += float(penalty)
                            samples += group
                            batch_rows += [row] * len(group)
                        active = retry
                        if algo != "dapo":
                            break
                    metrics.update({"sampling/attempts": attempts, "sampling/degenerate_groups": degenerate})
                if not samples:
                    logger.log(
                        step,
                        {**metrics, "update/skipped": 1.0, "sampling/seconds": time.perf_counter() - started},
                    )
                    continue
                lp, mask, values = policy.score(samples, values=algo == "ppo")
                old = padded_old(samples, policy.device)
                rewards = torch.tensor([s.reward for s in samples], device=policy.device)
                metrics["reward/mean"] = rewards.mean()
                ref_lp = None
                if reference is not None:
                    with torch.no_grad():
                        ref_lp, _, _ = reference.score(samples)
                if algo in {"opd", "opsd", "sar-opd", "idt-opd"}:
                    current_teacher = reference if general_phase else teacher
                    tlp = teacher_logprobs(
                        current_teacher, policy, samples, batch_rows, "solution" if algo == "opsd" else None
                    )
                    loss, kl = opd_loss(lp, old, tlp, mask)
                    metrics["distill/sampled_reverse_kl"] = kl
                elif algo == "ppo":
                    with torch.no_grad():
                        old_values = values.detach()
                        token_rewards = -config.get("beta", 0.02) * (old - ref_lp) * mask
                        for i, s in enumerate(samples):
                            last = max(j for j, m in enumerate(s.mask[1:]) if m)
                            token_rewards[i, last] += s.reward
                        advantages, returns = gae(
                            token_rewards, old_values, mask, lam=config.get("gae_lambda", 0.95)
                        )
                        mean = masked_mean(advantages, mask)
                        std = masked_mean((advantages - mean).square(), mask).sqrt()
                        advantages = (advantages - mean) / (std + 1e-8) * mask
                    loss, stats = ppo_loss(lp, old, values, old_values, advantages, returns, mask)
                    metrics.update(stats)
                else:
                    if algo == "tempo":
                        advantages = torch.nn.utils.rnn.pad_sequence(
                            [torch.tensor(s.advantages[1:], device=policy.device) for s in samples],
                            batch_first=True,
                        )
                    else:
                        advantages = group_advantages(
                            rewards, config["group_size"], scale=config.get("scale_rewards", True)
                        )
                    if algo == "agentopsd":
                        # Current parameters are unchanged since rollout; skill-conditioned teacher is this snapshot.
                        tlp = teacher_logprobs(policy, policy, samples, batch_rows, "skill")
                        turn_adv = torch.zeros_like(lp)
                        for i, (sample, a) in enumerate(zip(samples, advantages)):
                            evidence = [
                                float((tlp[i, start - 1 : end - 1] - old[i, start - 1 : end - 1]).sum())
                                for start, end in sample.turns
                            ]
                            group_start = (i // config["group_size"]) * config["group_size"]
                            base = float(rewards[group_start : group_start + config["group_size"]].mean())
                            credit = reshape_turn_advantages(float(a), base, evidence)
                            for (start, end), turn in zip(sample.turns, credit.turns):
                                turn_adv[i, start - 1 : end - 1] = turn.advantage
                            sample.metadata["turn_evidence"] = evidence
                            sample.metadata["turn_advantages"] = [x.advantage for x in credit.turns]
                        advantages = turn_adv
                    if config.get("mask_truncated", False):
                        for i, sample in enumerate(samples):
                            if sample.metadata.get("truncated"):
                                mask[i] = 0
                    kind = (
                        "gspo" if algo == "gspo" else config.get("loss", "dapo" if algo == "dapo" else "grpo")
                    )
                    loss, stats = policy_loss(
                        lp,
                        old,
                        advantages,
                        mask,
                        kind=kind,
                        clip_low=config.get("clip_low", 0.2),
                        clip_high=config.get("clip_high", 0.2),
                        ref_logp=ref_lp,
                        beta=config.get("beta", 0.0),
                        max_length=config["max_new_tokens"],
                    )
                    metrics.update(stats)
                accelerator.backward(loss)
                metrics["loss/total"] = loss.detach()
                metrics["tokens/trainable"] = mask.sum()
            norm = accelerator.clip_grad_norm_(policy.parameters(), config.get("max_grad_norm", 1.0))
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient; refusing optimizer update")
            optimizer.step()
            metrics.update(
                {
                    "update/grad_norm": norm,
                    "update/skipped": float(norm == 0),
                    "time/elapsed_seconds": time.perf_counter() - started,
                }
            )
            logger.log(step, metrics)
            logger.trajectories(step, [s.record() for s in samples])
            if (step + 1) % config.get("save_every", 10) == 0:
                policy.save(logger.path / f"checkpoint-{step + 1}")
        policy.save(logger.path / "checkpoint-final")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": config["steps"],
                "rng_state": torch.get_rng_state(),
            },
            logger.path / "training-state.pt",
        )
        (logger.path / "status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "device": str(policy.device),
                    "steps_requested": config["steps"],
                    "benchmark_validated": False,
                }
            )
            + "\n"
        )
        return logger.path
    finally:
        logger.close()
