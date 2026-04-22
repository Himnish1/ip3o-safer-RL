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