"""Mechanism tests against the installed verl API, without any GPU allocation."""

import math

import pytest
import torch

pytest.importorskip("verl")
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from agentic_rl.models import Sample
from agentic_rl.verl_backend.protocol import pack, unpack, pad_for_world
from agentic_rl.verl_backend.losses import REGISTERED
from agentic_rl.verl_backend.tempo import ReplayStore, prefix_correction
from agentic_rl.verl_backend.gpu_config import make_gpu_config
from agentic_rl.cli import load_config


@pytest.mark.asyncio
async def test_gpu_replicas_sleep_before_first_weight_sync(monkeypatch):
    from types import SimpleNamespace
    from agentic_rl.verl_backend import gpu_rollout

    calls = []

    async def sleep():
        calls.append("sleep")

    async def create(config, worker_group):
        calls.append("create")
        return SimpleNamespace(
            rollout_replicas=[SimpleNamespace(sleep=sleep)],
            server_addresses=["address"],
            server_handles=["handle"],
            global_load_balancer="balancer",
        )

    def client(config, servers, balancer):
        assert calls == ["create", "sleep"]
        calls.append("client")
        return "client"

    monkeypatch.setattr(gpu_rollout._ServersOnlyManager, "create", create)
    monkeypatch.setattr(gpu_rollout, "AsyncLLMServerManager", client)
    manager = SimpleNamespace(config={}, runtime=SimpleNamespace(group="group"))
    await gpu_rollout.GPUGenerationManager._initialize(manager)
    assert calls == ["create", "sleep", "client"]


@pytest.mark.asyncio
async def test_gpu_mode_transitions_await_existing_event_loop():
    from types import SimpleNamespace
    from agentic_rl.verl_backend.gpu_worker import GPUWorker

    calls = []

    async def rollout_mode():
        calls.append("synchronize_and_wake")

    async def release():
        calls.append("release_inference_weights_and_cache")

    worker = SimpleNamespace(version=3, rollout_mode=rollout_mode, rollout=SimpleNamespace(release=release))
    assert await GPUWorker.enter_rollout(worker) == 3
    await GPUWorker.enter_train(worker)
    assert calls == ["synchronize_and_wake", "release_inference_weights_and_cache"]


def test_exact_token_roundtrip_and_zero_weight_dispatch_padding():
    s = Sample(
        [1, 4, 8, 9, 12, 13],
        [0, 0, 1, 0, 0, 1],
        2,
        old_logp=[0, -0.4, 0, 0, -0.9],
        turns=[(2, 3), (5, 6)],
        metadata={"observations": ["tool"]},
    )
    batch = pack([s])
    restored = unpack(batch)[0]
    assert restored.tokens == s.tokens and restored.old_logp == s.old_logp
    assert restored.turns == s.turns and restored.mask == s.mask
    padded, count = pad_for_world(batch, 2)
    assert count == 1
    assert padded.batch["response_mask"][0].sum() == 2
    assert padded.batch["response_mask"][1].sum() == 0
    assert batch.batch["response_mask"].sum() == 2


@pytest.mark.parametrize(
    "kind,reduction", [("grpo", "sequence"), ("gspo", "sequence"), ("dapo", "token"), ("opd", "token")]
)
def test_global_normalization_matches_unsplit_gradient(kind, reduction):
    assert kind in REGISTERED
    old = torch.full((3, 5), -2.0)
    new = (old + torch.tensor([[0.1] * 5, [-0.1] * 5, [0.05] * 5])).requires_grad_()
    advantage = torch.tensor([[1.0] * 5, [-0.8] * 5, [0.3] * 5])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0.0]])
    common = {"global_tokens": 8, "global_sequences": 3, "reduction": reduction}
    kernel = get_policy_loss_fn("lab_" + kind)
    full = kernel(old, new, advantage, mask, config={**common, "world_size": 1})[0]
    expected = torch.autograd.grad(full, new)[0]
    pieces = []
    for start, end in ((0, 2), (2, 3)):
        piece = kernel(
            old[start:end],
            new[start:end],
            advantage[start:end],
            mask[start:end],
            config={**common, "world_size": 2},
        )[0]
        pieces.append(piece)
    actual = torch.autograd.grad(sum(pieces) / 2, new)[0]
    torch.testing.assert_close(actual, expected)


def test_opd_registered_gradient_uses_frozen_teacher_gap():
    old = torch.tensor([[-2.0, -3.0]])
    teacher = torch.tensor([[-1.0, -4.0]], requires_grad=True)
    new = old.clone().requires_grad_()
    advantage = (teacher - old).detach()
    loss = get_policy_loss_fn("lab_opd")(
        old,
        new,
        advantage,
        torch.ones_like(old),
        config={"global_tokens": 2, "global_sequences": 1, "reduction": "token"},
    )[0]
    loss.backward()
    assert new.grad[0, 0] < 0 and new.grad[0, 1] > 0
    assert teacher.grad is None


def test_prefix_importance_ignores_tools_and_supports_explicit_clipping():
    state = {"tokens": [1, 2, 3, 4], "behavior_mask": [0, 1, 0, 1], "behavior_logp": [-2.0, -10.0, -3.0]}

    class Policy:
        def score(self, samples):
            return torch.tensor([[-1.5, 99.0, -2.5]]), None, None

    weight, log_ratio = prefix_correction(Policy(), state, {})
    assert log_ratio == pytest.approx(1.0) and weight == pytest.approx(math.e)
    assert prefix_correction(Policy(), state, {"prefix_is_clip": 2})[0] == pytest.approx(2)


def test_replay_preserves_behavior_across_multiple_policy_versions(tmp_path):
    row = {"id": "x", "prompt": "x"}
    first = Sample(
        [1, 2, 3],
        [0, 1, 0],
        1,
        old_logp=[-1.0, 0.0],
        metadata={"stop_reason": "macro_boundary", "actions": ["a"], "observations": ["o"]},
    )
    store = ReplayStore(seed=7)
    store.add(row, first)
    parent = store.pop()
    second = Sample(
        [1, 2, 3, 4, 5],
        [0, 0, 0, 1, 0],
        3,
        old_logp=[0, 0, -2.0, 0],
        metadata={"stop_reason": "macro_boundary", "actions": ["a", "b"], "observations": ["o", "p"]},
    )
    store.add(row, second, parent)
    saved = store.state_dict()
    clone = ReplayStore(seed=123)
    clone.load_state_dict(saved)
    state = clone.pop()
    assert state["behavior_mask"] == [0, 1, 0, 1, 0]
    assert state["behavior_logp"] == [-1.0, 0, -2.0, 0]
    assert state["depth"] == 2
    second.metadata["stop_reason"] = "context_limit"
    clone.add(row, second, parent)
    assert not clone.states


@pytest.mark.parametrize(
    "chapter",
    [
        "01-grpo",
        "02-opd",
        "03-search-r1",
        "04-opsd",
        "05-retool",
        "06-dapo",
        "07-gspo",
        "08-alfworld",
        "09-AgentOPSD",
        "09-tempo",
        "09-vision-grpo",
        "001-ppo",
        "12-harness-rl",
    ],
)
def test_gpu_config_composes_installed_schema_without_device_execution(chapter):
    from verl.utils.config import omega_conf_to_dataclass

    config = make_gpu_config(load_config(chapter + "/config.yaml"))
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    assert rollout.engine_kwargs["vllm"]["seed"] == 42
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.rollout.mode == "async"
    assert config.actor_rollout_ref.rollout.calculate_log_probs
    assert config.actor_rollout_ref.actor.fsdp_config.use_orig_params is False
    assert "hf_model" in config.actor_rollout_ref.actor.checkpoint.save_contents
