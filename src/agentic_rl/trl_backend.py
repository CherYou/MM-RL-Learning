"""Version-pinned, real TRL trainer entry points (0.25.1).

PPO uses a deterministic task verifier behind TRL's reward-model interface. A
pretrained learned reward model can be supplied instead. DAPO here exposes the
TRL loss ablation; full dynamic sampling and length shaping use native DAPO.
"""

import copy
import json
from types import SimpleNamespace
import torch
from torch import nn
from datasets import Dataset
from transformers import TrainerCallback, AutoModelForSequenceClassification, GPT2ForSequenceClassification
from trl import GRPOConfig, GRPOTrainer, DPOConfig, DPOTrainer, PPOConfig, PPOTrainer, SFTConfig, SFTTrainer
from .data import ROOT, read_jsonl
from .models import Policy
from .logging import RunLogger
from .rewards import exact_match
from .rollout import MATH_SYSTEM


class LocalCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        numeric = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
        if numeric:
            self.logger.log(state.global_step, numeric)


class TokenBackbone(nn.Module):
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=(input_ids.float().unsqueeze(-1),))


class VerifierHead(nn.Module):
    def __init__(self, tokenizer, rows, debug=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.rows = rows
        self.debug = debug

    def forward(self, hidden):
        result = torch.zeros_like(hidden)
        for i, tokens in enumerate(hidden[:, :, 0].long()):
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if self.debug:
                value = float((tokens % 7 == 0).sum())
            else:
                matches = [r for r in self.rows if r["prompt"] in text]
                if not matches:
                    raise ValueError(
                        "PPO verifier cannot match prompt; increase context limit and retain full prompt"
                    )
                row = max(matches, key=lambda r: len(r["prompt"]))
                response = text.split(row["prompt"], 1)[1]
                value = exact_match(response, row.get("answers") or row["answer"])
            result[i, :, 0] = value
        return result


class VerifierRewardModel(nn.Module):
    base_model_prefix = "backbone"

    def __init__(self, tokenizer, rows, debug=False):
        super().__init__()
        self.backbone = TokenBackbone()
        self.score = VerifierHead(tokenizer, rows, debug)


def run_trl(config):
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    policy = Policy(config["model"], config["device"], config["seed"], config.get("lora_rank", 0))
    tok = policy.tokenizer
    tok.padding_side = "left"
    rows = read_jsonl(config["dataset"])
    evaluation = read_jsonl(config["eval_dataset"])
    if {r["prompt"] for r in rows} & {r["prompt"] for r in evaluation}:
        raise ValueError("Train/eval prompt overlap")
    logger = RunLogger(ROOT / config["output"], config)
    common = dict(
        output_dir=str(logger.path / "trainer"),
        learning_rate=config["learning_rate"],
        max_steps=config["steps"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=1,
        use_cpu=config["device"] == "cpu",
        bf16=False,
        fp16=False,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        seed=config["seed"],
        data_seed=config["seed"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    callbacks = [LocalCallback(logger)]
    algo = config["algorithm"]
    try:
        if algo == "dpo":
            args = DPOConfig(
                **common,
                beta=config.get("dpo_beta", 0.1),
                max_length=config.get("max_context_tokens", 4096),
                max_completion_length=config["max_new_tokens"],
            )

            def cols(source):
                return Dataset.from_list(
                    [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in source]
                )

            trainer = DPOTrainer(
                model=policy.lm,
                ref_model=copy.deepcopy(policy.lm),
                processing_class=tok,
                args=args,
                train_dataset=cols(rows),
                eval_dataset=cols(evaluation),
                callbacks=callbacks,
            )
        elif algo == "ppo":
            common.pop("max_steps")
            args = PPOConfig(
                **common,
                total_episodes=config["steps"] * config["batch_size"],
                response_length=config["max_new_tokens"],
                num_ppo_epochs=config.get("ppo_epochs", 2),
                num_mini_batches=1,
                num_sample_generations=0,
                local_rollout_forward_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                temperature=1.0,
                whiten_rewards=False,
                stop_token="eos",
                kl_coef=config.get("beta", 0.02),
            )

            def encode(source):
                return Dataset.from_list(
                    [{"input_ids": policy.prompt(r["prompt"], MATH_SYSTEM)[0]} for r in source]
                )

            if config["model"] == "tiny":
                vc = copy.deepcopy(policy.lm.config)
                vc.num_labels = 1
                value = GPT2ForSequenceClassification(vc)
                value.transformer.load_state_dict(policy.lm.transformer.state_dict())
            else:
                value = AutoModelForSequenceClassification.from_pretrained(
                    config["model"], num_labels=1, torch_dtype=torch.float32, ignore_mismatched_sizes=True
                )
            value.config.pad_token_id = tok.pad_token_id
            rm = (
                AutoModelForSequenceClassification.from_pretrained(config["reward_model"], num_labels=1)
                if config.get("reward_model")
                else VerifierRewardModel(tok, rows + evaluation, config.get("reward") == "debug_token")
            )
            trainer = PPOTrainer(
                args=args,
                model=policy.lm,
                ref_model=copy.deepcopy(policy.lm),
                value_model=value,
                reward_model=rm,
                processing_class=tok,
                train_dataset=encode(rows),
                eval_dataset=encode(evaluation),
                callbacks=callbacks,
            )
        elif algo == "sft":

            def supervised(source):
                return Dataset.from_list(
                    [{"prompt": r["prompt"], "completion": r["solution"]} for r in source]
                )

            args = SFTConfig(
                **common, max_length=config.get("max_context_tokens", 4096), completion_only_loss=True
            )
            trainer = SFTTrainer(
                model=policy.lm,
                processing_class=tok,
                args=args,
                train_dataset=supervised(rows),
                eval_dataset=supervised(evaluation),
                callbacks=callbacks,
            )
        elif algo in {"grpo", "gspo", "dapo"}:
            common["per_device_train_batch_size"] = config["batch_size"] * config["group_size"]
            args = GRPOConfig(
                **common,
                num_generations=config["group_size"],
                max_completion_length=config["max_new_tokens"],
                max_prompt_length=config.get("max_prompt_tokens", 1024),
                use_vllm=False,
                loss_type="dapo" if algo == "dapo" else config.get("loss", "grpo"),
                importance_sampling_level="sequence" if algo == "gspo" else "token",
                epsilon=config.get("clip_low", 0.2),
                epsilon_high=config.get("clip_high", 0.2),
                beta=config.get("beta", 0.0),
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                scale_rewards="group",
                mask_truncated_completions=config.get("mask_truncated", False),
            )

            def verifier(completions, answer, **kwargs):
                texts = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
                if config.get("reward") == "debug_token":
                    return [float(sum(ord(ch) % 7 == 0 for ch in text)) for text in texts]
                return [exact_match(text, gold) for text, gold in zip(texts, answer)]

            def dataset(source):
                return Dataset.from_list(
                    [
                        {
                            "prompt": tok.decode(policy.prompt(r["prompt"], MATH_SYSTEM)[0]),
                            "answer": r["answer"],
                        }
                        for r in source
                    ]
                )

            trainer = GRPOTrainer(
                model=policy.lm,
                processing_class=tok,
                args=args,
                reward_funcs=verifier,
                train_dataset=dataset(rows),
                eval_dataset=dataset(evaluation),
                callbacks=callbacks,
            )
        else:
            raise ValueError(f"{algo} uses the native PyTorch/Accelerate backend")
        trainer.train()
        trainer.save_model(str(logger.path / "checkpoint-final/model"))
        tok.save_pretrained(logger.path / "checkpoint-final/model")
        trainer.save_state()
        if algo == "ppo":
            value.save_pretrained(logger.path / "checkpoint-final/value_model")
        (logger.path / "status.json").write_text(
            json.dumps({"status": "completed", "backend": "trl", "benchmark_validated": False}) + "\n"
        )
        return logger.path
    finally:
        logger.close()
