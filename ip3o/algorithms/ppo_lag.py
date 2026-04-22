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
        self.ac = actor_critic
        self.cfg = config
        self.device = device
        self.lagrange_multiplier = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.policy_lr)

    def _policy_loss(self, batch):
        logp, entropy, _, _ = self.ac.evaluate_actions(batch["obs"], batch["actions"])
        ratio = torch.exp(logp - batch["log_probs"])
        combined_adv = batch["reward_adv"] - self.lagrange_multiplier.detach() * batch["cost_adv"]
        clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * combined_adv
        policy_loss = -(torch.min(ratio * combined_adv, clipped)).mean() - self.cfg.entropy_coef * entropy.mean()
        approx_kl = (batch["log_probs"] - logp).mean().abs()
        return policy_loss, approx_kl

    def _value_loss(self, batch):
        _, _, v_reward, v_cost = self.ac.evaluate_actions(batch["obs"], batch["actions"])
        return F.mse_loss(v_reward, batch["reward_returns"]) + F.mse_loss(v_cost, batch["cost_returns"])

    def update_lagrange_multiplier(self, mean_cost_return: torch.Tensor) -> None:
        violation = mean_cost_return.detach() - self.cfg.cost_limit
        self.lagrange_multiplier = torch.clamp(
            self.lagrange_multiplier + self.cfg.lambda_lr * violation,
            min=0.0,
        )

    def update(self, batch):
        info = {}
        for _ in range(self.cfg.train_iters):
            self.optimizer.zero_grad()
            policy_loss, approx_kl = self._policy_loss(batch)
            value_loss = self._value_loss(batch)
            total_loss = policy_loss + value_loss
            total_loss.backward()
            self.optimizer.step()
            info["kl"] = float(approx_kl.item())
            if approx_kl > self.cfg.target_kl:
                break

        self.update_lagrange_multiplier(batch["cost_returns"].mean())
        info.update(
            {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "lagrange_multiplier": float(self.lagrange_multiplier.item()),
            }
        )
        return info
