"""Registered verl AgentLoop for exact-token mathematical/tool/household rollouts.

Environment adapters run in threads, while generate requests are coalesced by an
async manager into real worker batches. This keeps tool waits out of model code.
"""

import asyncio
from dataclasses import asdict
import time
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    DictConfigWrap,
    register,
)
from verl.workers.rollout.replica import TokenOutput
from verl.utils.dataset.rl_dataset import RLHFDataset

from agentic_rl.models import Sample
from agentic_rl.rollout import single_rollout, agent_rollout
from agentic_rl.harness import harness_rollout, CallRecord


class BatchServerManager:
    def __init__(self, policy, max_batch=64, delay=0.002):
        self.policy, self.max_batch, self.delay = policy, max_batch, delay
        self.pending = []
        self.task = None

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = {
            "ids": prompt_ids,
            "max_new_tokens": sampling_params["max_new_tokens"],
            "greedy": sampling_params.get("greedy", False),
            "extras": image_data or {},
        }
        self.pending.append((request, future))
        if self.task is None:
            self.task = asyncio.create_task(self._flush())
        return await future

    async def _flush(self):
        try:
            while self.pending:
                await asyncio.sleep(self.delay)
                items, self.pending = self.pending[: self.max_batch], self.pending[self.max_batch :]
                try:
                    outputs = await asyncio.to_thread(self.policy.generate_many, [x[0] for x in items], True)
                    for (_, future), (tokens, log_probs) in zip(items, outputs, strict=True):
                        if not future.done():
                            future.set_result(TokenOutput(token_ids=tokens, log_probs=log_probs))
                except BaseException as error:
                    for _, future in items:
                        if not future.done():
                            future.set_exception(error)
        finally:
            self.task = None


class AsyncPolicyBridge:
    def __init__(self, policy, manager, event_loop, request_id):
        self.policy, self.manager, self.event_loop, self.request_id = policy, manager, event_loop, request_id

    def __getattr__(self, key):
        return getattr(self.policy, key)

    def generate(self, prompt_ids, max_new_tokens, extras=None, greedy=False):
        coroutine = self.manager.generate(
            self.request_id,
            prompt_ids=prompt_ids,
            sampling_params={"max_new_tokens": max_new_tokens, "greedy": greedy},
            image_data=extras,
        )
        result = asyncio.run_coroutine_threadsafe(coroutine, self.event_loop).result(timeout=600)
        return result.token_ids


@register("agentic_rl_lab")
class LabAgentLoop(AgentLoopBase):
    def __init__(self, *args, lab_config=None, policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lab_config = lab_config or OmegaConf.to_container(self.config.lab, resolve=True)
        self.policy = policy or self.server_manager.policy

    async def run(self, sampling_params, **kwargs):
        start = time.perf_counter()
        request_id = kwargs["request_id"]
        bridge = AsyncPolicyBridge(self.policy, self.server_manager, asyncio.get_running_loop(), request_id)
        row = kwargs["row"]
        records = None
        if self.lab_config["algorithm"] == "harness-rl":
            records, _ = await asyncio.to_thread(harness_rollout, bridge, row, self.lab_config, request_id)
            if not records:
                raise ValueError("Harness context budget leaves no interface call")
            sample = records[-1].sample
        elif self.lab_config["algorithm"] in {"search-r1", "retool", "alfworld", "agentopsd", "tempo"}:
            sample = await asyncio.to_thread(
                agent_rollout,
                bridge,
                row,
                self.lab_config,
                initial=kwargs.get("initial"),
                turn_limit=kwargs.get("turn_limit"),
            )
        else:
            sample = await asyncio.to_thread(single_rollout, bridge, row, self.lab_config)
        prefix = sample.prompt_length
        return AgentLoopOutput(
            prompt_ids=sample.tokens[:prefix],
            response_ids=sample.tokens[prefix:],
            response_mask=[int(m) for m in sample.mask[prefix:]],
            response_logprobs=sample.old_logp[prefix - 1 :],
            reward_score=sample.reward,
            num_turns=len(sample.turns),
            metrics=AgentLoopMetrics(generate_sequences=time.perf_counter() - start),
            extra_fields={
                "sample": asdict(sample),
                "records": [asdict(r) for r in records] if records else None,
            },
        )


async def collect_async(policy, rows, config, *, step=0, initial=None, turn_limit=None):
    manager = BatchServerManager(policy, max_batch=config.get("rollout_batch_size", 64))
    trainer_cfg = OmegaConf.create({"actor_rollout_ref": {"rollout": {}, "model": {}}, "lab": config})
    data_cfg = OmegaConf.create({})
    calls = []
    for i, row in enumerate(rows):
        loop = LabAgentLoop(
            DictConfigWrap(trainer_cfg),
            manager,
            policy.tokenizer,
            None,
            RLHFDataset,
            DictConfigWrap(data_cfg),
            lab_config=config,
            policy=policy,
        )
        calls.append(loop.run({}, row=row, request_id=f"{step}-{i}", initial=initial, turn_limit=turn_limit))
    return await asyncio.gather(*calls)


def collect(policy, rows, config, **kwargs):
    outputs = asyncio.run(collect_async(policy, rows, config, **kwargs))
    samples = [Sample(**out.extra_fields["sample"]) for out in outputs]
    records = []
    for out in outputs:
        group = []
        for record in out.extra_fields["records"] or []:
            record["sample"] = Sample(**record["sample"])
            group.append(CallRecord(**record))
        records.append(group)
    return samples, records
