from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPOLagConfig:
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_iters: int = 10
    target_kl: float = 0.015
    entropy_coef: float = 0.0
    cost_limit: float = 25.0
    lambda_lr: float = 0.05


class PPOLagrangian:
    def __init__(self, actor_critic, config: PPOLagConfig, device: torch.device):
        pass