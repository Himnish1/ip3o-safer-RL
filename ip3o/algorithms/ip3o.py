from __future__ import annotations

from dataclasses import dataclass

import torch

from ip3o.algorithms.ppo_lag import PPOLagConfig, PPOLagrangian


@dataclass
class IP3OConfig(PPOLagConfig):
    initial_penalty: float = 0.1
    penalty_increment: float = 0.05
    max_penalty: float = 10.0


class IP3O(PPOLagrangian):
    """Incrementally penalized PPO with logarithmic barrier term."""

    def __init__(self, actor_critic, config: IP3OConfig, device: torch.device):
        super().__init__(actor_critic, config, device)