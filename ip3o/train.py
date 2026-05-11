from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import yaml

from ip3o.algorithms.ip3o import IP3O, IP3OConfig
from ip3o.algorithms.ppo_lag import PPOLagConfig, PPOLagrangian
from ip3o.env.wrappers import make_safety_point_goal
from ip3o.models.actor_critic import ActorCritic
from ip3o.utils.buffer import RolloutBuffer
from ip3o.utils.logger import Logger

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent(algorithm: str, actor_critic, config: dict, device: torch.device):
    if algorithm == "ip3o":
        return IP3O(actor_critic, IP3OConfig(**config), device), asdict(IP3OConfig(**config))
    if algorithm == "ppo_lag":
        return PPOLagrangian(actor_critic, PPOLagConfig(**config), device), asdict(PPOLagConfig(**config))
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train(config_path: str):
    cfg = load_config(config_path)
    device = pick_device()

    env = make_safety_point_goal(cfg.get("env_id", "SafetyPointGoal1-v0"))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac = ActorCritic(obs_dim, act_dim, hidden_sizes=cfg.get("hidden_sizes", [128, 128])).to(device)
    agent, effective_cfg = build_agent(cfg["algorithm"], ac, cfg["algorithm_config"], device)

    buffer = RolloutBuffer(gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95))
    logger = Logger()

    obs, _ = env.reset(seed=cfg.get("seed", 0))
    for epoch in range(cfg.get("epochs", 1)):
        for _ in range(cfg.get("steps_per_epoch", 128)):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t, vr_t, vc_t = ac.act(obs_t)
            action = action_t.squeeze(0).cpu().numpy()
            step = env.step(action)
            done = step.terminated or step.truncated
            buffer.add(obs, action, step.reward, step.cost, done, logp_t.item(), vr_t.item(), vc_t.item())
            obs = step.observation
            if done:
                obs, _ = env.reset()

        batch = buffer.get(device)
        info = agent.update(batch)
        logger.log(epoch=epoch, mean_reward=batch["reward_returns"].mean().item(), mean_cost=batch["cost_returns"].mean().item(), **info)
        buffer.clear()

    logger.to_csv(cfg.get("log_path", "ip3o_logs/train_metrics.csv"))
    return logger.latest(), effective_cfg, str(device)


def main():
    parser = argparse.ArgumentParser(description="Train PPO-Lag or IP3O on Safety-Gymnasium")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    latest, cfg, device = train(args.config)
    print({"device": device, "config": cfg, "latest": latest})


if __name__ == "__main__":
    main()