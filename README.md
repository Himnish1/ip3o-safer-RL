# ip3o-safer-RL

This repository now includes a minimal, from-scratch IP3O (Incrementally Penalized PPO) implementation for Safety-Gymnasium `SafetyPointGoal1-v0` using PyTorch, with automatic MPS selection when available.

## Project structure

```
ip3o/
├── env/
│   └── wrappers.py
├── models/
│   └── actor_critic.py
├── algorithms/
│   ├── ppo_lag.py
│   └── ip3o.py
├── utils/
│   ├── buffer.py
│   └── logger.py
├── train.py
└── configs/
    ├── ppo_lag.yaml
    └── ip3o.yaml
tests/
└── test_ip3o_core.py
```

## Step-by-step: run IP3O

1. Install dependencies:
   - `pip install torch pyyaml safety-gymnasium`
2. Train PPO-Lagrangian baseline:
   - `python -m ip3o.train --config ip3o/configs/ppo_lag.yaml`
3. Train IP3O:
   - `python -m ip3o.train --config ip3o/configs/ip3o.yaml`
4. Compare output CSV logs:
   - `ip3o_logs/ppo_lag_metrics.csv`
   - `ip3o_logs/ip3o_metrics.csv`

## What is implemented

- Dual-critic actor-critic network (`V_R` and `V_C`) with shared MLP backbone.
- PPO-Lagrangian baseline with adaptive Lagrange multiplier updates.
- IP3O variant with logarithmic barrier penalty and incremental penalty annealing.
- Safety-Gymnasium wrapper that normalizes cost extraction from environment `info`.
