"""Transformers policies with exact token preservation across tool turns."""

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast


@dataclass
class Sample:
    tokens: list[int]
    mask: list[float]
    prompt_length: int
    old_logp: list[float] = field(default_factory=list)
    reward: float = 0.0
    text: str = ""
    turns: list[tuple[int, int]] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    advantages: list[float] | None = None

    def validate(self):
        if len(self.tokens) != len(self.mask) or len(self.old_logp) != len(self.tokens) - 1:
            raise ValueError("Token/mask/logprob alignment mismatch")
        if any(self.mask[: self.prompt_length]):
            raise ValueError("Prompt tokens cannot be trained")
        if not any(self.mask):
            raise ValueError("Empty trainable response")

    def record(self):
        return {
            "text": self.text,
            "reward": self.reward,
            "token_ids": self.tokens,
            "loss_mask": self.mask,
            "old_logprobs": self.old_logp,
            "turn_spans": self.turns,
            "metadata": self.metadata,
        }


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {s: i for i, s in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"] + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=4096,
    )
    tok.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return tok


class Policy(nn.Module):
    def __init__(self, model_name="tiny", device="cpu", seed=42, lora_rank=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model_name = model_name
        self.tokenizer = (
            tiny_tokenizer() if model_name == "tiny" else AutoTokenizer.from_pretrained(model_name)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_name == "tiny":
            config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=4096,
                n_embd=64,
                n_layer=2,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
            self.lm = GPT2LMHeadModel(config)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, attn_implementation="eager"
            )
        if lora_rank:
            from peft import LoraConfig, get_peft_model

            self.lm = get_peft_model(
                self.lm,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                    lora_dropout=0.0,
                ),
            )
        self.value_head = nn.Linear(self.lm.config.hidden_size, 1)
        self.to(device)
        # Eval mode disables dropout without disabling gradients; rollout ratios stay meaningful.
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, text, system="", image_path=None):
        if image_path:
            raise ValueError("Use VisionPolicy for image-conditioned rollouts")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": text}
        ]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True), {}
        return self.encode((system + "\n" if system else "") + "User: " + text + "\nAssistant: "), {}

    def score(self, samples, values=False):
        max_len = max(len(s.tokens) for s in samples)
        ids = torch.full(
            (len(samples), max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
        )
        attention = torch.zeros_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        for i, sample in enumerate(samples):
            ids[i, : len(sample.tokens)] = torch.tensor(sample.tokens, device=self.device)
            attention[i, : len(sample.tokens)] = 1
            mask[i, : len(sample.tokens)] = torch.tensor(sample.mask, device=self.device)
        outputs = self.lm(
            input_ids=ids, attention_mask=attention, output_hidden_states=values, use_cache=False
        )
        logp = outputs.logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        predicted_values = self.value_head(outputs.hidden_states[-1][:, :-1]).squeeze(-1) if values else None
        return logp, mask[:, 1:], predicted_values

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, extras=None, greedy=False):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        if ids.shape[1] + max_new_tokens > self.lm.config.max_position_embeddings:
            raise ValueError("Context budget exceeded; reduce prompt/turn/token limits explicitly")
        kwargs = {"do_sample": not greedy}
        if not greedy:
            kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
        result = self.lm.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
            **(extras or {}),
        )
        return result[0, len(prompt_ids) :].tolist()

    @torch.no_grad()
    def snapshot(self, sample):
        lp, _, _ = self.score([sample])
        sample.old_logp = lp[0].cpu().tolist()
        sample.validate()
        return sample

    def frozen_copy(self):
        result = copy.deepcopy(self).eval()
        result.requires_grad_(False)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(directory / "model")
        self.tokenizer.save_pretrained(directory / "model")
        torch.save(self.value_head.state_dict(), directory / "value_head.pt")
        (directory / "policy.json").write_text(json.dumps({"model_name": self.model_name}) + "\n")


class VisionPolicy(Policy):
    """Standard Transformers LLaVA for offline smoke, Auto VLM for real experiments.

    The tiny checkpoint is randomly initialized and serves only to verify that
    pixels condition logprobs and gradients. Real configs can use Qwen2.5-VL.
    """

    def __init__(self, model_name="tiny", device="cpu", seed=42, lora_rank=0):
        nn.Module.__init__(self)
        from transformers import AutoProcessor, AutoModelForImageTextToText

        torch.manual_seed(seed)
        self.model_name = model_name
        checkpoint_meta = Path(model_name).parent / "policy.json"
        tiny_checkpoint = (
            checkpoint_meta.is_file() and json.loads(checkpoint_meta.read_text()).get("model_name") == "tiny"
        )
        if model_name == "tiny" or tiny_checkpoint:
            from transformers import LlavaConfig, LlavaForConditionalGeneration

            self.tokenizer = tiny_tokenizer()
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
            self.image_id = self.tokenizer.convert_tokens_to_ids("<image>")
            config = LlavaConfig(
                vision_config={
                    "model_type": "clip_vision_model",
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "image_size": 16,
                    "patch_size": 8,
                },
                text_config={
                    "model_type": "llama",
                    "vocab_size": len(self.tokenizer),
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 4096,
                    "pad_token_id": 0,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                },
                image_token_index=self.image_id,
                image_seq_length=4,
                vision_feature_layer=-1,
            )
            self.lm = LlavaForConditionalGeneration(config)
            self.processor = None
            if tiny_checkpoint:
                self.lm = LlavaForConditionalGeneration.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model_name = "tiny"
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.processor.tokenizer
            self.lm = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.float32, attn_implementation="eager"
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if lora_rank:
            from peft import LoraConfig, get_peft_model

            self.lm = get_peft_model(
                self.lm,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=2 * lora_rank,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
        self.value_head = nn.Linear(self.lm.config.text_config.hidden_size, 1)
        self.blocked_output_ids = sorted(
            {
                v
                for key in ("image_token_index", "image_token_id", "video_token_id")
                if (v := getattr(self.lm.config, key, None)) is not None
            }
        )
        self.to(device).eval()

    def prompt(self, text, system="", image_path=None):
        from PIL import Image
        import numpy as np

        if not image_path:
            raise ValueError("Image path is required")
        with Image.open(image_path) as file:
            picture = file.convert("RGB")
        if self.processor is None:
            ids = [self.image_id] * 4 + self.encode(system + "\n" + text + "\nAssistant: ")
            pixels = (
                torch.tensor(np.asarray(picture.resize((16, 16))).copy(), dtype=torch.float32).permute(
                    2, 0, 1
                )[None]
                / 255
            )
            return ids, {"pixel_values": pixels.to(self.device)}
        messages = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
        ]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[rendered], images=[picture], return_tensors="pt").to(self.device)
        return inputs.pop("input_ids")[0].tolist(), {k: v for k, v in inputs.items() if k != "attention_mask"}

    def score(self, samples, values=False):
        outputs = []
        masks = []
        for s in samples:
            ids = torch.tensor([s.tokens], device=self.device)
            out = self.lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **s.extras)
            logits = out.logits[0, :-1].float().clone()
            logits[:, self.blocked_output_ids] = -torch.inf
            logp = logits.log_softmax(-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
            # Image markers only occur in context; zero them before a masked reduction.
            active = torch.tensor(s.mask[1:], device=self.device).bool()
            outputs.append(torch.where(active, logp, torch.zeros_like(logp)))
            masks.append(torch.tensor(s.mask[1:], device=self.device))
        return (
            nn.utils.rnn.pad_sequence(outputs, batch_first=True),
            nn.utils.rnn.pad_sequence(masks, batch_first=True),
            None,
        )

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, extras=None, greedy=False):
        ids = torch.tensor([prompt_ids], device=self.device)
        kwargs = {} if greedy else {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
        out = self.lm.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            suppress_tokens=self.blocked_output_ids,
            **kwargs,
            **(extras or {}),
        )
        return out[0, len(prompt_ids) :].tolist()

    def save(self, directory):
        super().save(directory)
        if self.processor is not None:
            self.processor.save_pretrained(Path(directory) / "model")
