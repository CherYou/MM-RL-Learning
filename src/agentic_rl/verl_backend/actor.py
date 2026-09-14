"""verl DataParallelPPOActor extension for distillation, temporal credit and CAPO."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import get_policy_loss_fn

from . import losses
from .protocol import unpack


class LabPPOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer, lab_config, *, local_policy=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.lab_config = lab_config
        self.local_policy = local_policy
        self.partitions = None

    @property
    def device(self):
        if self.local_policy is None:
            from verl.utils.device import get_device_name

            return torch.device(get_device_name())
        return next(self.actor_module.parameters()).device

    def forward_data(self, data, *, values=False):
        data = data.to(self.device)
        if self.local_policy is not None:
            lp, _, value = self.local_policy.score(unpack(data, self.device), values=values)
            # Preserve the global padded width after dispatch/microbatch splitting.
            width = data.batch["responses"].shape[-1]
            lp = torch.nn.functional.pad(lp, (0, width - lp.shape[-1]))
            if value is not None:
                value = torch.nn.functional.pad(value, (0, width - value.shape[-1]))
            return {"log_probs": lp, "values": value}
        inputs = {**data.batch, **data.non_tensor_batch}
        if any(bool(x) for x in inputs.get("multi_modal_inputs", [])):
            outputs = []
            for i, extras in enumerate(inputs["multi_modal_inputs"]):
                length = int(inputs["attention_mask"][i].sum())
                ids = inputs["input_ids"][i : i + 1, :length]
                extras = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in extras.items()
                    if not k.startswith("_")
                }
                with torch.autocast(device_type=self.device.type, dtype=self.param_dtype):
                    result = self.actor_module(
                        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **extras
                    )
                logits = result.logits[0, :-1].float().clone()
                model_config = self.actor_module.config
                blocked = sorted(
                    {
                        v
                        for k in ("image_token_index", "image_token_id", "video_token_id")
                        if (v := getattr(model_config, k, None)) is not None
                    }
                )
                logits[:, blocked] = -torch.inf
                lp = logits.log_softmax(-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
                active = inputs["response_mask"][i, : length - 1].bool()
                lp = torch.where(active, lp, torch.zeros_like(lp))
                outputs.append(torch.nn.functional.pad(lp, (0, inputs["responses"].shape[-1] - len(lp))))
            return {"log_probs": torch.stack(outputs)}
        # Empty multimodal dicts must not trigger VLM-specific concatenation.
        if not any(bool(x) for x in inputs.get("multi_modal_inputs", [])):
            inputs.pop("multi_modal_inputs", None)
        return self._forward_micro_batch(inputs, temperature=data.meta_info["temperature"])

    @torch.no_grad()
    def infer(self, data, *, values=False):
        self.actor_module.eval()
        outputs = []
        for micro in data.split(self.lab_config.get("micro_batch_size", 1)):
            outputs.append(self.forward_data(micro, values=values))
        result = {
            key: torch.cat([out[key] for out in outputs]).detach().cpu()
            for key in outputs[0]
            if outputs[0][key] is not None
        }
        # FSDP keeps the root full parameters after a no-grad forward. Restore
        # its local shard before the GPU worker offloads it, as upstream does.
        if isinstance(self.actor_module, FSDP) and dist.get_world_size() > 1:
            handle = self.actor_module._handle
            if handle is not None:
                handle.reshard(True)
        return result

    def _synchronize_unsharded_gradients(self):
        if not isinstance(self.actor_module, FSDP) and dist.get_world_size() > 1:
            # The CPU reference worker uses the same verl dispatch as FSDP workers.
            # No DDP wrapper is bypassed by Policy.score: synchronization is explicit.
            for parameter in self.actor_module.parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= dist.get_world_size()

    def update_policy(self, data):
        self.actor_module.eval()  # dropout must match rollout; autograd stays enabled
        cfg = {**self.lab_config, **data.meta_info["loss_config"]}
        name = data.meta_info["loss_kind"]
        kernel = get_policy_loss_fn("lab_" + name)
        epochs = cfg.get("ppo_epochs", 1)
        stats = {}
        totals = []
        for _ in range(epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)
            channels = ("action", "args") if self.partitions is not None else ("policy",)
            action_gradients = {}
            for channel in channels:
                if channel == "args":
                    self.actor_optimizer.zero_grad(set_to_none=True)
                for micro in data.split(cfg.get("micro_batch_size", 1)):
                    micro = micro.to(self.device)
                    outputs = self.forward_data(micro, values=cfg["algorithm"] == "ppo")
                    lp, batch = outputs["log_probs"], micro.batch
                    mask_key = "response_mask" if channel == "policy" else channel + "_mask"
                    mask = batch[mask_key]
                    channel_cfg = dict(cfg)
                    if channel != "policy":
                        channel_cfg.update(
                            global_tokens=cfg[channel + "_tokens"], reduction="token", beta=0.0
                        )
                    loss, measured = kernel(
                        old_log_prob=batch["old_log_probs"],
                        log_prob=lp,
                        advantages=batch["advantages"],
                        response_mask=mask,
                        config=channel_cfg,
                        rollout_is_weights=batch.get("prefix_is_weights"),
                    )
                    # PPO KL is part of token reward, not added a second time here.
                    if cfg["algorithm"] != "ppo":
                        loss, kl = losses.add_reference_kl(
                            loss, lp, batch.get("ref_log_prob"), mask, channel_cfg
                        )
                        measured["policy/kl"] = kl
                    value = outputs.get("values")
                    if cfg["algorithm"] == "ppo" and value is not None:
                        old_v, returns = batch["old_values"], batch["returns"]
                        clipped = old_v + (value - old_v).clamp(
                            -cfg.get("value_clip", 0.2), cfg.get("value_clip", 0.2)
                        )
                        vl = 0.5 * losses.reduce_tokens(
                            torch.maximum((value - returns).square(), (clipped - returns).square()),
                            mask,
                            channel_cfg,
                            kind="token",
                        )
                        loss = loss + cfg.get("value_coef", 0.5) * vl
                        measured["loss/value"] = float(vl.detach())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite verl loss")
                    loss.backward()
                    totals.append(float(loss.detach()))
                    stats.update(measured)
                self._synchronize_unsharded_gradients()
                if channel == "action":
                    action_gradients = {
                        n: p.grad.detach().clone()
                        for n, p in self.actor_module.named_parameters()
                        if p.grad is not None
                    }
            if self.partitions is not None:
                routed = self.partitions.route(self.actor_module, action_gradients)
                if isinstance(self.actor_module, FSDP):
                    count = torch.tensor(routed, device=self.device)
                    dist.all_reduce(count)
                    routed = int(count)
                stats["capo/routed_parameters"] = routed
            norm = self._optimizer_step()  # real verl clipping and optimizer update
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite verl gradient")
            stats["update/grad_norm"] = float(norm)
            stats["update/skipped"] = float(norm == 0)
        stats["loss/total"] = sum(totals) / epochs
        stats["tokens/trainable"] = cfg["global_tokens"]
        stats["verl/optimizer_steps"] = epochs
        self.actor_optimizer.zero_grad(set_to_none=True)
        return stats
