#!/usr/bin/env python3
"""Exercise real two-rank CPU FSDP + verl actor + CAPO sharded gradient routing."""

import hashlib
import json
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

from agentic_rl.data import ROOT
from agentic_rl.models import Policy
from agentic_rl.harness import load_probes
from agentic_rl.verl_backend.actor import LabPPOActor
from agentic_rl.verl_backend.worker import actor_config
from agentic_rl.verl_backend.capo import PartitionBank
from agentic_rl.verl_backend.protocol import pack
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from omegaconf import OmegaConf


def worker(rank, path):
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method="file://" + path + "/store", rank=rank, world_size=2)
    try:
        policy = Policy("tiny", "cpu", 42)
        policy.lm = FSDP(policy.lm, device_id=torch.device("cpu"), use_orig_params=False)
        cfg = {"algorithm": "harness-rl", "micro_batch_size": 1, "ppo_epochs": 1, "learning_rate": 1e-3}
        optimizer = torch.optim.AdamW(policy.lm.parameters(), lr=cfg["learning_rate"], weight_decay=0.0)
        actor = LabPPOActor(actor_config(cfg), policy.lm, optimizer, cfg)
        probes = load_probes(policy, ROOT / "data/harness/probes.jsonl")
        bank = PartitionBank(0.25)
        bank.probe(policy.lm, probes, policy.score)
        actor.partitions = bank
        samples = [r.sample for r in probes]
        for sample in samples:
            sample.advantages = [1.0 * m for m in sample.mask]
        data = pack(samples, policy.tokenizer.pad_token_id)
        data.batch["action_mask"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(r.action_mask[1:]) for r in probes], batch_first=True
        )
        data.batch["args_mask"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(r.args_mask[1:]) for r in probes], batch_first=True
        )
        data.meta_info.update(
            loss_kind="dapo",
            loss_config={
                "world_size": 2,
                "global_sequences": 2,
                "global_tokens": float(data.batch["response_mask"].sum()),
                "action_tokens": float(data.batch["action_mask"].sum()),
                "args_tokens": float(data.batch["args_mask"].sum()),
                "reduction": "token",
            },
        )

        def state():
            with FSDP.state_dict_type(
                policy.lm,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                return {k: v.detach().clone() for k, v in policy.lm.state_dict().items()}

        before = state()
        inference = actor.infer(data.chunk(2)[rank])
        local = data.chunk(2)[rank].batch
        valid = local["response_mask"].bool()
        # Upstream computes arbitrary logprobs at padding positions; only
        # actual assistant tokens belong to the behavior-policy comparison.
        assert torch.allclose(inference["log_probs"][valid], local["old_log_probs"][valid], atol=1e-5)
        metrics = actor.update_policy(data.chunk(2)[rank])
        after = state()
        changed = 0
        for name, prior in before.items():
            diff = after[name] - prior
            a = bank._mask_slice(name, prior.shape, 0, prior.numel(), 0, "cpu").reshape_as(prior)
            g = bank._mask_slice(name, prior.shape, 0, prior.numel(), 1, "cpu").reshape_as(prior)
            assert torch.count_nonzero(diff[(a + g) == 0]) == 0, name
            changed += int(torch.count_nonzero(diff))
        assert changed > 0
        digest = hashlib.sha256(b"".join(t.cpu().numpy().tobytes() for t in after.values())).hexdigest()
        digests = [None, None]
        dist.all_gather_object(digests, digest)
        assert digests[0] == digests[1]
        manager = FSDPCheckpointManager(
            model=policy.lm,
            optimizer=optimizer,
            processing_class=policy.tokenizer,
            checkpoint_config=OmegaConf.create(
                {
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                }
            ),
        )
        manager.save_checkpoint(path + "/checkpoint", global_step=1)
        with torch.no_grad():
            for parameter in policy.lm.parameters():
                parameter.zero_()
        manager.load_checkpoint(path + "/checkpoint", del_local_after_load=False)
        restored = state()
        assert all(torch.equal(restored[k], v) for k, v in after.items())
        if rank == 0:
            (ROOT / "reports/verl-fsdp-capo.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "device": "cpu",
                        "ranks": 2,
                        "sharding": "FULL_SHARD",
                        "actor_base": "verl.workers.actor.dp_actor.DataParallelPPOActor",
                        "parameters_changed": changed,
                        "unselected_parameters_changed": 0,
                        "rank_parameters_equal": True,
                        "fsdp_inference_reshard": True,
                        "official_checkpoint_restore_equal": True,
                        "metrics": metrics,
                    },
                    indent=2,
                )
                + "\n"
            )
            print("REAL_CPU_FSDP_CAPO_PASSED", changed, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="arl-fsdp-") as path:
        mp.spawn(worker, args=(path,), nprocs=2, join=True)
