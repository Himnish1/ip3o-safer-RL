from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def _build_mlp(sizes: Iterable[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers = []
    sizes = list(sizes)
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Iterable[int] = (128, 128)):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.backbone = _build_mlp([obs_dim, *hidden_sizes], activation=nn.Tanh)
        self.actor_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.reward_value = nn.Linear(hidden_sizes[-1], 1)
        self.cost_value = nn.Linear(hidden_sizes[-1], 1)

    def _features(self, obs: Tensor) -> Tensor:
        return self.backbone(obs)

    def distribution(self, obs: Tensor) -> Normal:
        features = self._features(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def values(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        features = self._features(obs)
        return self.reward_value(features).squeeze(-1), self.cost_value(features).squeeze(-1)

    def act(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        v_reward, v_cost = self.values(obs)
        return action, log_prob, v_reward, v_cost

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        v_reward, v_cost = self.values(obs)
        return log_prob, entropy, v_reward, v_cost