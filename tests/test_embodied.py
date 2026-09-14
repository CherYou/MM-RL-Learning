"""Mathematical and data-boundary contracts for the continuous-control lessons."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from agentic_rl.embodied.agents import Agent, bellman_target, expectile_loss
from agentic_rl.embodied.networks import SquashedPolicy
from agentic_rl.embodied.replay import Replay, relabel_future, transition
from agentic_rl.embodied.runner import validate_config


def batch():
    return (
        torch.randn(8, 13),
        torch.rand(8, 4) * 1.8 - 0.9,
        -torch.ones(8),
        torch.randn(8, 13),
        torch.zeros(8),
    )


def vector(module):
    return torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()


def test_time_limit_bootstrap_differs_from_absorbing_terminal():
    result = bellman_target(
        torch.tensor([-1.0, -1.0]), torch.tensor([0.0, 1.0]), torch.tensor([5.0, 5.0]), 0.9
    )
    torch.testing.assert_close(result, torch.tensor([3.5, -1.0]))


def test_upper_expectile_optimum_for_two_recorded_values():
    # With equally frequent Q=0 and Q=2, tau=.75 yields V=1.5, not the mean 1.
    value = torch.tensor(1.5, requires_grad=True)
    loss = expectile_loss(torch.tensor([0.0, 2.0]) - value, 0.75)
    loss.backward()
    assert loss.item() == pytest.approx(0.375)
    assert value.grad.item() == pytest.approx(0.0, abs=1e-7)


def test_squashed_density_matches_change_of_variables_and_saturated_gradients():
    distribution = torch.distributions.Normal(torch.zeros(2, 4), torch.ones(2, 4))
    raw = torch.tensor([[0.1, -0.2, 0.3, -0.4], [20.0, -20.0, 15.0, -15.0]], requires_grad=True)
    actual = SquashedPolicy.corrected_log_prob(distribution, raw)
    expected = (distribution.log_prob(raw)[:1] - torch.log1p(-raw[:1].tanh().square())).sum(-1)
    torch.testing.assert_close(actual[:1], expected)
    actual.sum().backward()
    assert torch.isfinite(actual).all() and torch.isfinite(raw.grad).all()


def test_td3_delays_actor_and_targets_but_updates_critics():
    torch.manual_seed(2)
    agent = Agent({"algorithm": "td3", "hidden_dim": 16, "policy_delay": 2})
    actor, critic, target = vector(agent.actor), vector(agent.critic), vector(agent.target_critic)
    first = agent.update(batch())
    assert first["train/actor_updated"] == 0
    assert torch.equal(actor, vector(agent.actor)) and torch.equal(target, vector(agent.target_critic))
    assert not torch.equal(critic, vector(agent.critic))
    second = agent.update(batch())
    assert second["train/actor_updated"] == 1
    assert not torch.equal(actor, vector(agent.actor)) and not torch.equal(
        target, vector(agent.target_critic)
    )


def test_sac_reparameterizes_and_updates_temperature_without_target_gradients():
    torch.manual_seed(3)
    agent = Agent({"algorithm": "sac", "hidden_dim": 16})
    actor, alpha = vector(agent.actor), agent.log_alpha.detach().clone()
    metrics = agent.update(batch())
    assert not torch.equal(actor, vector(agent.actor))
    assert not torch.equal(alpha, agent.log_alpha.detach())
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(parameter.grad is None for parameter in agent.target_critic.parameters())


def test_iql_never_queries_q_on_policy_sampled_actions(monkeypatch):
    torch.manual_seed(4)
    agent = Agent({"algorithm": "iql", "hidden_dim": 16})
    data = batch()
    seen = []
    for network in (agent.critic, agent.target_critic):
        network.register_forward_pre_hook(lambda module, args: seen.append(args[1].detach().clone()))

    def forbidden(*args, **kwargs):
        raise AssertionError("IQL should train using the recorded action")

    monkeypatch.setattr(agent.actor, "sample", forbidden)
    initial = vector(agent.actor)
    metrics = agent.update(data)
    assert seen and all(torch.equal(actions, data[1]) for actions in seen)
    assert not torch.equal(initial, vector(agent.actor))
    assert all(np.isfinite(value) for value in metrics.values())


def example_episode():
    rows = []
    for index in range(3):
        observation = {
            "observation": np.zeros(10),
            "achieved_goal": np.array([index, 0, 0]),
            "desired_goal": np.array([9, 0, 0]),
        }
        following = {**observation, "achieved_goal": np.array([index + 1, 0, 0])}
        rows.append(transition(observation, np.zeros(4), -1, following, False, index == 2, 7, index))
    return rows


def sparse_reward(achieved, goal, info):
    return -float(np.linalg.norm(achieved - goal) > 0.05)


def test_her_changes_both_goal_fields_and_reward_without_changing_history():
    episode = example_episode()
    hindsight = relabel_future(episode, sparse_reward, np.random.default_rng(1), 4)
    assert len(hindsight) == 12
    assert all(row["reward"] == -1 and row["desired_goal"][0] == 9 for row in episode)
    for row in hindsight:
        original = episode[row["time"]]
        assert row["future_time"] > row["time"] and row["future_time"] <= 3
        np.testing.assert_array_equal(row["desired_goal"], row["next_desired_goal"])
        assert row["desired_goal"][0] == row["future_time"]
        assert row["reward"] == sparse_reward(row["next_achieved_goal"], row["desired_goal"], {})
        for key in ("action", "observation", "next_observation", "achieved_goal", "next_achieved_goal"):
            np.testing.assert_array_equal(row[key], original[key])
    assert all(row["reward"] == 0 for row in hindsight[-4:])


@pytest.mark.parametrize("corruption", ["episode", "boundary", "order"])
def test_her_rejects_cross_episode_or_invalid_future_order(corruption):
    episode = deepcopy(example_episode())
    if corruption == "episode":
        episode[1]["episode"] = 8
    elif corruption == "boundary":
        episode[0]["truncated"] = True
    else:
        episode.reverse()
    with pytest.raises(ValueError):
        relabel_future(episode, sparse_reward, np.random.default_rng(1))


def test_replay_keeps_original_experience_and_timeouts_bootstrap():
    replay = Replay(100, 5)
    replay.add_episode(example_episode(), reward_function=sparse_reward, her_k=2)
    assert len(replay) == 9
    assert sum(not row["relabeled"] for row in replay.rows) == 3
    assert replay.sample(16)[-1].sum() == 0


def test_independent_evaluation_seed_guard():
    config = {
        "algorithm": "td3",
        "backend": "embodied",
        "device": "cpu",
        "seed": 42,
        "steps": 300,
        "eval_seed": 100,
        "eval_episodes": 3,
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_config(config)


@pytest.mark.parametrize("kind", ["sac", "td3", "her", "iql"])
def test_checkpoint_restores_actor_optimizer_and_continuation(kind, tmp_path):
    torch.manual_seed(5)
    config = {"algorithm": kind, "hidden_dim": 16}
    agent = Agent(config)
    data = batch()
    agent.update(data)
    agent.update(data)
    path = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), path)
    restored = Agent(config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(vector(agent.actor), vector(restored.actor))
    state = torch.get_rng_state()
    expected = agent.update(data)
    torch.set_rng_state(state)
    actual = restored.update(data)
    assert actual == expected
    assert torch.equal(vector(agent.actor), vector(restored.actor))
