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
        self.cfg: IP3OConfig = config
        self.penalty_coeff = torch.tensor(config.initial_penalty, dtype=torch.float32, device=device)

    def anneal_penalty(self) -> None:
        self.penalty_coeff = torch.clamp(self.penalty_coeff + self.cfg.penalty_increment, max=self.cfg.max_penalty)

    def incremental_penalty(self, mean_cost_return: torch.Tensor) -> torch.Tensor:
        safe_gap = torch.clamp(self.cfg.cost_limit - mean_cost_return, min=1e-6)
        normalized_gap = safe_gap / max(self.cfg.cost_limit, 1e-6)
        return self.penalty_coeff * (-torch.log(normalized_gap))

    def _policy_loss(self, batch):
        base_loss, approx_kl = super()._policy_loss(batch)
        penalty = self.incremental_penalty(batch["cost_returns"].mean())
        return base_loss + penalty, approx_kl

    def update(self, batch):
        info = super().update(batch)
        self.anneal_penalty()
        info["penalty_coeff"] = float(self.penalty_coeff.item())
        return info
