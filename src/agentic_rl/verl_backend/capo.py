"""CAPO partitions that can route ordinary and FSDP1 flattened/sharded gradients.

Partitions store unit indices, not dense model-sized masks. FSDP gradients are
routed after each backward/reduce-scatter, so action and argument contributions
can be separated without unsupported autograd.grad calls on a sharded model.
"""

import math
import torch


def canonical(name):
    name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
    if name.startswith("lm."):
        name = name[3:]
    return name


class PartitionBank:
    def __init__(self, fraction=0.25):
        if not 0 < fraction <= 1:
            raise ValueError("CAPO fraction must be in (0,1]")
        self.fraction = fraction
        self.specs = {}

    @torch.no_grad()
    def probe(self, model, records, score):
        modules = {canonical(n): m for n, m in model.named_modules() if n.endswith((".c_fc", ".up_proj"))}
        stats = {name: [None, None] for name in modules}
        eligible = [r for r in records if r.sample.reward > 0 and any(r.action_mask) and any(r.args_mask)]
        if not eligible or not modules:
            raise ValueError("CAPO needs successful structured probes and supported GPT2/Llama/Qwen MLPs")
        for record in eligible:
            handles = []
            for name, module in modules.items():

                def hook(_module, _inputs, output, name=name):
                    activation = output[0, :-1].float().clamp_min(0)
                    for channel, mask in enumerate((record.action_mask, record.args_mask)):
                        weight = torch.tensor(mask[1:], device=activation.device)[:, None]
                        contribution = (activation * weight).sum(0) / weight.sum().clamp_min(1)
                        prior = stats[name][channel]
                        stats[name][channel] = contribution if prior is None else prior + contribution

                handles.append(module.register_forward_hook(hook))
            try:
                score([record.sample])
            finally:
                for handle in handles:
                    handle.remove()
        for name, (action, args) in stats.items():
            units = action.numel()
            k = max(1, math.ceil(units * self.fraction))
            selection = []
            for score_vector in (action, args):
                mask = torch.zeros(units, dtype=torch.bool)
                mask[score_vector.topk(k).indices.cpu()] = True
                selection.append(mask)
            up_axis = 1 if name.endswith(".c_fc") else 0
            down = name.rsplit(".", 1)[0] + (".c_proj.weight" if up_axis == 1 else ".down_proj.weight")
            self.specs[name + ".weight"] = {"axis": up_axis, "units": selection}
            self.specs[name + ".bias"] = {"axis": 0, "units": selection}
            self.specs[down] = {"axis": 1 - up_axis, "units": selection}

    def _mask_slice(self, name, shape, start, count, channel, device):
        spec = self.specs.get(canonical(name))
        if spec is None:
            return torch.zeros(count, device=device)
        axis = spec["axis"]
        stride = math.prod(shape[axis + 1 :])
        units = (torch.arange(start, start + count, device=device) // stride) % shape[axis]
        return spec["units"][channel].to(device)[units]

    def mask(self, name, parameter, channel):
        if hasattr(parameter, "_param_infos"):
            result = torch.zeros_like(parameter.grad)
            prefix = name.rsplit(".", 1)[0] if "." in name else ""
            for info, shard, shape in zip(
                parameter._param_infos, parameter._shard_param_infos, parameter._shapes, strict=True
            ):
                if not shard.in_shard:
                    continue
                fqn = ".".join(x for x in (prefix, info.module_name, info.param_name) if x)
                start, count = shard.intra_param_start_idx, shard.numel_in_shard
                offset = shard.offset_in_shard
                result[offset : offset + count] = self._mask_slice(
                    fqn, shape, start, count, channel, result.device
                )
            return result
        return self._mask_slice(
            name, parameter.shape, 0, parameter.numel(), channel, parameter.device
        ).reshape_as(parameter)

    def route(self, model, action_gradients):
        selected = 0
        for name, parameter in model.named_parameters():
            a = action_gradients.get(name)
            g = parameter.grad
            if a is None and g is None:
                continue
            if g is None:
                parameter.grad = torch.zeros_like(a)
                g = parameter.grad
            ma, mg = self.mask(name, parameter, 0), self.mask(name, parameter, 1)
            parameter.grad = mg * g + (ma * a if a is not None else 0)
            selected += int(((ma + mg) > 0).sum())
        return selected

    def state_dict(self):
        return {"fraction": self.fraction, "specs": self.specs}

    def load_state_dict(self, state):
        self.fraction, self.specs = state["fraction"], state["specs"]
