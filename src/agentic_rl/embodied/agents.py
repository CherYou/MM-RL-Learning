"""SAC, TD3 and IQL from their objectives, sharing only network/optimizer plumbing.

HER is a replay transformation in replay.py, with TD3 as its base learner.
These are low-dimensional teaching agents, not pretrained VLA models.
"""

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .networks import DeterministicPolicy, SquashedPolicy, TwinQ, mlp


@torch.no_grad()
def polyak(source, target, tau):
    for current, slow in zip(source.parameters(), target.parameters(), strict=True):
        slow.lerp_(current, tau)


def expectile_loss(residual, expectile):
    weights = torch.where(residual > 0, expectile, 1 - expectile)
    return (weights * residual.square()).mean()


def bellman_target(rewards, terminated, next_values, discount):
    # A time limit is an episode boundary, but not an absorbing task state.
    return rewards + discount * (1 - terminated) * next_values


class Agent:
    def __init__(self, config, observation_dim=13, action_dim=4):
        self.config = dict(config)
        self.kind = config["algorithm"]
        self.discount = config.get("discount", 0.98)
        self.tau = config.get("tau", 0.005)
        self.updates = 0
        width = config.get("hidden_dim", 128)
        policy_type = DeterministicPolicy if self.kind in {"td3", "her"} else SquashedPolicy
        self.actor = policy_type(observation_dim, action_dim, width)
        self.critic = TwinQ(observation_dim, action_dim, width)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.value = mlp(observation_dim, 1, width)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.get("alpha", 0.2))))
        self.target_entropy = config.get("target_entropy", -float(action_dim))
        learning_rate = config.get("learning_rate", 3e-4)
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=learning_rate),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=learning_rate),
            "value": torch.optim.Adam(self.value.parameters(), lr=learning_rate),
            "alpha": torch.optim.Adam([self.log_alpha], lr=learning_rate),
        }

    @torch.no_grad()
    def act(self, state, *, explore=False, rng=None):
        states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if explore and self.kind == "sac":
            action = self.actor.sample(states)[0][0].numpy()
        else:
            action = self.actor(states)[0].numpy()
            if explore:
                if rng is None:
                    raise ValueError("Exploration requires an explicitly seeded generator")
                action += rng.normal(0, self.config.get("exploration_noise", 0.1), action.shape)
        return np.clip(action, -1, 1).astype(np.float32)

    def optimize(self, name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {name} loss")
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def update(self, batch):
        self.updates += 1
        if self.kind == "iql":
            return self.update_iql(batch)
        return self.update_online(batch)

    def update_online(self, batch):
        states, actions, rewards, following, terminated = batch
        stochastic = self.kind == "sac"
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            if stochastic:
                next_actions, next_log_prob = self.actor.sample(following)
                values = self.target_critic.minimum(following, next_actions) - alpha * next_log_prob
            else:
                noise = torch.randn_like(actions) * self.config.get("target_noise", 0.2)
                noise = noise.clamp(-self.config.get("noise_clip", 0.5), self.config.get("noise_clip", 0.5))
                next_actions = (self.target_actor(following) + noise).clamp(-1, 1)
                values = self.target_critic.minimum(following, next_actions)
            target = bellman_target(rewards, terminated, values, self.discount)
        first, second = self.critic(states, actions)
        metrics = {
            "loss/critic": self.optimize("critic", F.mse_loss(first, target) + F.mse_loss(second, target))
        }
        actor_due = stochastic or self.updates % self.config.get("policy_delay", 2) == 0
        metrics["train/actor_updated"] = float(actor_due)
        if actor_due:
            # Freeze Q weights, while keeping dQ/da for the actor's chain rule.
            self.critic.requires_grad_(False)
            if stochastic:
                proposed, log_prob = self.actor.sample(states)
                actor_loss = (alpha * log_prob - self.critic.minimum(states, proposed)).mean()
            else:
                actor_loss = -self.critic(states, self.actor(states))[0].mean()
            metrics["loss/actor"] = self.optimize("actor", actor_loss)
            self.critic.requires_grad_(True)
            if stochastic and self.config.get("learn_alpha", True):
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                metrics["loss/alpha"] = self.optimize("alpha", alpha_loss)
            polyak(self.critic, self.target_critic, self.tau)
            if not stochastic:
                polyak(self.actor, self.target_actor, self.tau)
        metrics.update({"train/q": float(first.detach().mean()), "train/target": float(target.mean())})
        if stochastic:
            metrics["train/alpha"] = float(self.log_alpha.detach().exp())
        return metrics

    def update_iql(self, batch):
        states, actions, rewards, following, terminated = batch
        with torch.no_grad():
            recorded_q = self.target_critic.minimum(states, actions)
        values = self.value(states).squeeze(-1)
        value_loss = expectile_loss(recorded_q - values, self.config.get("expectile", 0.7))
        metrics = {"loss/value": self.optimize("value", value_loss)}
        with torch.no_grad():
            target = bellman_target(rewards, terminated, self.value(following).squeeze(-1), self.discount)
            advantages = recorded_q - self.value(states).squeeze(-1)
            # Clamp the exponent before exp so even a very large advantage is safe.
            log_weights = self.config.get("advantage_scale", 3.0) * advantages
            weights = log_weights.clamp(max=math.log(self.config.get("max_weight", 100.0))).exp()
        first, second = self.critic(states, actions)
        metrics["loss/critic"] = self.optimize(
            "critic", F.mse_loss(first, target) + F.mse_loss(second, target)
        )
        actor_loss = -(weights * self.actor.log_prob(states, actions)).mean()
        metrics["loss/actor"] = self.optimize("actor", actor_loss)
        polyak(self.critic, self.target_critic, self.tau)
        metrics.update({"train/advantage": float(advantages.mean()), "train/weight": float(weights.mean())})
        return metrics

    def state_dict(self):
        return {
            "config": self.config,
            "updates": self.updates,
            "networks": {
                name: getattr(self, name).state_dict()
                for name in ("actor", "critic", "target_actor", "target_critic", "value")
            },
            "log_alpha": self.log_alpha.detach(),
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
        }

    def load_state_dict(self, state):
        if state["config"]["algorithm"] != self.kind:
            raise ValueError("Checkpoint algorithm differs from requested algorithm")
        for name, weights in state["networks"].items():
            getattr(self, name).load_state_dict(weights)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        for name, weights in state["optimizers"].items():
            self.optimizers[name].load_state_dict(weights)
        self.updates = state["updates"]
