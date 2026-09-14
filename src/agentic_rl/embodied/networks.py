"""Goal-conditioned MLPs; actions are normalized to [-1, 1]."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, outputs, width):
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, outputs),
    )


class TwinQ(nn.Module):
    def __init__(self, observations, actions, width):
        super().__init__()
        self.first = mlp(observations + actions, 1, width)
        self.second = mlp(observations + actions, 1, width)

    def forward(self, states, actions):
        joint = torch.cat((states, actions), dim=-1)
        return self.first(joint).squeeze(-1), self.second(joint).squeeze(-1)

    def minimum(self, states, actions):
        first, second = self(states, actions)
        return torch.minimum(first, second)


class DeterministicPolicy(nn.Module):
    def __init__(self, observations, actions, width):
        super().__init__()
        self.network = mlp(observations, actions, width)

    def forward(self, states):
        return self.network(states).tanh()


class SquashedPolicy(nn.Module):
    def __init__(self, observations, actions, width):
        super().__init__()
        self.network = mlp(observations, 2 * actions, width)

    def distribution(self, states):
        mean, log_std = self.network(states).chunk(2, dim=-1)
        return torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())

    @staticmethod
    def corrected_log_prob(distribution, raw):
        # log(1-tanh(u)^2), evaluated without cancellation near saturation.
        jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return (distribution.log_prob(raw) - jacobian).sum(dim=-1)

    def sample(self, states):
        distribution = self.distribution(states)
        raw = distribution.rsample()
        return raw.tanh(), self.corrected_log_prob(distribution, raw)

    def log_prob(self, states, recorded_actions):
        # Finite inverse for recorded actions at the actuator's clipping boundary.
        raw = torch.atanh(recorded_actions.clamp(-1 + 1e-6, 1 - 1e-6))
        return self.corrected_log_prob(self.distribution(states), raw)

    def forward(self, states):
        return self.distribution(states).mean.tanh()
