from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ip3o.algorithms.ppo_lag import PPOLagConfig, PPOLagrangian


@dataclass
class IP3OConfig(PPOLagConfig):
    eta: float = 0.5
    gamma: float = 0.99


class IP3O(PPOLagrangian):
    """Incrementally penalized PPO with fixed CELU barrier.

    The "incremental" penalty scaling comes from CELU's implicit escalation:
    when L_C < 0 (safe), CELU incentivizes staying safe.
    When L_C > 0 (unsafe), CELU linearly penalizes violations.
    As training progresses, L_C naturally increases, self-scaling the penalty.
    """

    def __init__(self, actor_critic, config: IP3OConfig, device: torch.device):
        super().__init__(actor_critic, config, device)
        self.cfg: IP3OConfig = config
        # Cached values from the last policy inner-step for accurate logging
        self._last_l_cost: float = 0.0
        self._last_celu: float = 0.0

    def _policy_loss(self, batch):
        """Compute IP3O policy loss with fixed CELU barrier.

        Loss = L_R + eta * CELU(L_C)
        Note: consolidate forward passes so the network is evaluated once per call.
        """
        # Single forward pass for policy-related quantities
        logp, entropy, _, _ = self.ac.evaluate_actions(batch["obs"], batch["actions"])
        ratio = torch.exp(logp - batch["log_probs"])

        # Reward loss (PPO with min clip)
        clipped_reward = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        reward_loss = -torch.min(ratio * batch["reward_adv"],
                                 clipped_reward * batch["reward_adv"]).mean()
        reward_loss -= self.cfg.entropy_coef * entropy.mean()
        approx_kl = (batch["log_probs"] - logp).mean().abs()

        # Cost surrogate (reuse the same ratio and clipping)
        # NOTE: for cost we use max(clip, no-clip) per paper
        cost_clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio)
        cost_surrogate = torch.max(ratio * batch["cost_adv"],
                                   cost_clipped * batch["cost_adv"]).mean()
        per_step_cost_limit = self.cfg.cost_limit * (1.0 - self.cfg.gamma)
        # = 25 * 0.01 = 0.25 per step

        l_cost = cost_surrogate + batch["cost_returns"].mean() - per_step_cost_limit

        celu_val = F.celu(l_cost)
        penalty = self.cfg.eta * celu_val

        # Cache the scalar values for logging after the inner loop finishes
        # Convert to Python floats to avoid holding graph references
        try:
            self._last_l_cost = float(l_cost.item())
            self._last_celu = float(celu_val.item())
        except Exception:
            # If conversion fails (e.g., distributed tensors), fall back safely
            self._last_l_cost = float(l_cost.detach().cpu().item())
            self._last_celu = float(celu_val.detach().cpu().item())

        return reward_loss + penalty, approx_kl

    def update(self, batch):
        info = super().update(batch)
        # Use cached values from the last inner iteration — do NOT recompute l_cost here
        info["l_cost"] = self._last_l_cost
        info["celu_penalty"] = self._last_celu
        print(f"  l_cost:       {info['l_cost']:.4f}")
        print(f"  celu_penalty: {info['celu_penalty']:.4f}")
        return info
