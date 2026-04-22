from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class SafetyStep:
    observation: Any
    reward: float
    cost: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class SafetyGymnasiumWrapper:
    """Normalizes Safety-Gymnasium step output to always expose scalar cost."""

    def __init__(self, env: Any):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> SafetyStep:
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(info.get("cost", info.get("cost_sum", 0.0)))
        return SafetyStep(obs, float(reward), cost, bool(terminated), bool(truncated), info)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_safety_point_goal(env_id: str = "SafetyPointGoal1-v0", **kwargs: Any) -> SafetyGymnasiumWrapper:
    try:
        import safety_gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "safety_gymnasium is required for SafetyPointGoal1-v0. Install with `pip install safety-gymnasium`."
        ) from exc

    env = gym.make(env_id, **kwargs)
    return SafetyGymnasiumWrapper(env)
