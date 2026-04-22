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
