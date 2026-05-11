from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ip3o.algorithms.ppo_lag import PPOLagConfig, PPOLagrangian


@dataclass
class IP3OConfig(PPOLagConfig):
    initial_penalty: float = 0.1
    penalty_increment: float = 0.05
    max_penalty: float = 10.0
    gamma: float = 0.99


class IP3O(PPOLagrangian):
    """Incrementally penalized PPO with logarithmic barrier term."""

    def __init__(self, actor_critic, config: IP3OConfig, device: torch.device):
        super().__init__(actor_critic, config, device)
        self.cfg: IP3OConfig = config
        self.penalty_coeff = torch.tensor(config.initial_penalty, dtype=torch.float32, device=device)

    def anneal_penalty(self) -> None:
        self.penalty_coeff = torch.clamp(self.penalty_coeff + self.cfg.penalty_increment, max=self.cfg.max_penalty)

    def incremental_penalty(self, l_cost: torch.Tensor) -> torch.Tensor:
        """Apply CELU barrier to the full cost loss term L_C.

        Args:
            l_cost: Full L_C term = (1/(1-gamma)) * cost_surrogate + J_C - d

        Returns:
            Penalty = eta * CELU(L_C)
        """
        return self.penalty_coeff * F.celu(l_cost)

    def _policy_loss(self, batch):
        """Compute IP3O policy loss with clipped cost surrogate.

        Loss = reward_loss + penalty_term

        Penalty term = eta * CELU(L_C) where
        L_C = (1/(1-gamma)) * E[max(clip(ratio) * A_C, ratio * A_C)] + J_C - d
        """
        logp, entropy, _, _ = self.ac.evaluate_actions(batch["obs"], batch["actions"])
        ratio = torch.exp(logp - batch["log_probs"])

        # Reward loss (PPO with min clip)
        clipped_reward = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        reward_loss = -torch.min(ratio * batch["reward_adv"],
                                 clipped_reward * batch["reward_adv"]).mean()
        reward_loss -= self.cfg.entropy_coef * entropy.mean()
        approx_kl = (batch["log_probs"] - logp).mean().abs()

        # Cost surrogate (PPO with max clip — opposite of reward)
        clipped_cost = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        cost_surrogate = torch.max(ratio * batch["cost_adv"],
                                   clipped_cost * batch["cost_adv"]).mean()

        # Full L_C = (1/(1-gamma)) * cost_surrogate + J_C - d
        l_cost = (1.0 / (1.0 - self.cfg.gamma)) * cost_surrogate \
                 + batch["cost_returns"].mean() \
                 - self.cfg.cost_limit

        penalty = self.incremental_penalty(l_cost)
        return reward_loss + penalty, approx_kl

    def update(self, batch):
        info = super().update(batch)
        self.anneal_penalty()
        info["penalty_coeff"] = float(self.penalty_coeff.item())
        return info
