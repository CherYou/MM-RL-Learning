"""Exact-token serialization for verl. No decode/encode round trip at the boundary.

The first token is the fixed one-token prefix; remaining tokens are responses.
Original prompt/tool/history tokens remain in response_ids with response_mask=0.
This permits variable prompts and non-contiguous trainable turns in one batch.
"""

from dataclasses import asdict
import numpy as np
import torch
from verl import DataProto
from agentic_rl.models import Sample


def object_array(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def pack(samples, pad_token_id=0, *, meta=None, tensors=None):
    if not samples:
        raise ValueError("Cannot send an empty trajectory batch")
    length = max(len(s.tokens) for s in samples)
    if length < 2:
        raise ValueError("A trajectory must have a context token and a predicted token")
    batch = len(samples)
    ids = torch.full((batch, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros(batch, length - 1)
    old = torch.zeros_like(mask)
    advantages = torch.zeros_like(mask)
    for i, sample in enumerate(samples):
        n = len(sample.tokens)
        if n != len(sample.mask) or any(sample.mask[: sample.prompt_length]):
            raise ValueError("Corrupt trajectory mask")
        ids[i, :n] = torch.tensor(sample.tokens)
        attention[i, :n] = 1
        mask[i, : n - 1] = torch.tensor(sample.mask[1:])
        if sample.old_logp:
            if len(sample.old_logp) != n - 1:
                raise ValueError("Old probabilities must align with exact sampled tokens")
            old[i, : n - 1] = torch.tensor(sample.old_logp)
        if sample.advantages is not None:
            advantages[i, : n - 1] = torch.tensor(sample.advantages[1:])
    values = dict(
        input_ids=ids,
        attention_mask=attention,
        position_ids=(attention.cumsum(-1) - 1).clamp_min(0),
        prompts=ids[:, :1],
        responses=ids[:, 1:],
        response_mask=mask,
        old_log_probs=old,
        advantages=advantages,
    )
    values.update(tensors or {})
    return DataProto.from_dict(
        tensors=values,
        non_tensors={
            "samples": object_array([to_cpu(asdict(s)) for s in samples]),
            "multi_modal_inputs": object_array([to_cpu(s.extras) for s in samples]),
        },
        meta_info={"temperature": 1.0, **(meta or {})},
    )


def unpack(data, device="cpu"):
    result = []
    for record in data.non_tensor_batch["samples"]:
        sample = Sample(**record)
        sample.extras = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.extras.items()
        }
        result.append(sample)
    return result


def pad_for_world(data, world_size):
    """Pad worker dispatch with zero-weight copies, never duplicate training weight."""
    extra = (-len(data)) % world_size
    if extra:
        dummy = data.select_idxs([0] * extra)
        dummy.batch = dummy.batch.clone()
        for key in ("response_mask", "advantages", "action_mask", "args_mask"):
            if key in dummy.batch:
                dummy.batch[key].zero_()
        data = DataProto.concat([data, dummy])
    return data, extra
