"""
compare_runs.py
---------------
Train both PPO-Lag and IP3O, collect per-epoch metrics, and plot side-by-side.

Usage:
    python compare_runs.py \
        --ppo_config ip3o/configs/ppo_lag.yaml \
        --ip3o_config ip3o/configs/ip3o.yaml \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ip3o.train import train  # returns (latest_info, effective_cfg, device_str)


# ---------------------------------------------------------------------------
# Run both algorithms and collect full epoch logs
# ---------------------------------------------------------------------------

def run_experiment(config_path: str, label: str) -> list[dict]:
    """Run training and return the logger's full history."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    latest, cfg, device = train(config_path)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s on {device}")
    return latest, cfg, elapsed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"PPO-Lagrangian": "#2196F3", "IP3O": "#F44336"}
METRICS = [
    ("mean_reward",          "Mean Reward Return",         False),
    ("mean_cost",            "Mean Cost Return",            True),   # True = draw cost_limit line
    ("policy_loss",          "Policy Loss",                 False),
    ("value_loss",           "Value Loss",                  False),
    ("kl",                   "Approx KL Divergence",        False),
]


def smooth(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Pad edges to avoid shrinkage
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def plot_comparison(
    ppo_history: list[dict],
    ip3o_history: list[dict],
    cost_limit: float,
    output_dir: Path,
) -> None:
    n_metrics = len(METRICS)
    fig = plt.figure(figsize=(16, 4 * n_metrics))
    gs = gridspec.GridSpec(n_metrics, 1, hspace=0.45)

    histories = {"PPO-Lagrangian": ppo_history, "IP3O": ip3o_history}

    for row, (key, title, add_limit_line) in enumerate(METRICS):
        ax = fig.add_subplot(gs[row])

        for label, history in histories.items():
            epochs = [d["epoch"] for d in history if key in d]
            values = [d[key] for d in history if key in d]
            if not values:
                continue
            smoothed = smooth(values)
            color = COLORS[label]
            ax.plot(epochs, values, alpha=0.25, color=color, linewidth=1)
            ax.plot(epochs, smoothed, label=label, color=color, linewidth=2)

        if add_limit_line:
            ax.axhline(cost_limit, color="black", linestyle="--",
                       linewidth=1.2, label=f"Cost limit ({cost_limit})")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Extra panel: penalty_coeff vs lagrange_multiplier
    ax_coeff = fig.add_subplot(gs[n_metrics - 1])  # reuse last slot or add below
    fig.set_size_inches(16, 4 * (n_metrics + 1))
    ax_coeff = fig.add_subplot(n_metrics + 1, 1, n_metrics + 1)

    for label, history in histories.items():
        color = COLORS[label]
        if label == "IP3O":
            epochs = [d["epoch"] for d in history if "penalty_coeff" in d]
            values = [d["penalty_coeff"] for d in history if "penalty_coeff" in d]
            ax_coeff.plot(epochs, values, label="IP3O penalty_coeff", color=color, linewidth=2)
        else:
            epochs = [d["epoch"] for d in history if "lagrange_multiplier" in d]
            values = [d["lagrange_multiplier"] for d in history if "lagrange_multiplier" in d]
            ax_coeff.plot(epochs, values, label="PPO-Lag λ", color=color, linewidth=2)

    ax_coeff.set_title("Constraint Coefficient Over Training", fontsize=13, fontweight="bold")
    ax_coeff.set_xlabel("Epoch")
    ax_coeff.legend(fontsize=9)
    ax_coeff.grid(True, alpha=0.3)
    ax_coeff.spines[["top", "right"]].set_visible(False)

    out_path = output_dir / "comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Patch train() to expose full history
# ---------------------------------------------------------------------------
# train() currently returns only `latest`. We need to intercept the Logger.
# Monkey-patch Logger.log to accumulate all records before train() returns.

from ip3o.utils import logger as _logger_module

_history_store: dict[str, list[dict]] = {}
_current_run_key: str = ""

_original_log = _logger_module.Logger.log

def _patched_log(self, **kwargs):
    _original_log(self, **kwargs)
    _history_store.setdefault(_current_run_key, []).append(dict(kwargs))

_logger_module.Logger.log = _patched_log


def run_and_collect(config_path: str, label: str):
    global _current_run_key
    _current_run_key = label
    _history_store[label] = []
    latest, cfg, elapsed = run_experiment(config_path, label)
    return _history_store[label], cfg, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_config",  default="ip3o/configs/ppo_lag.yaml")
    parser.add_argument("--ip3o_config", default="ip3o/configs/ip3o.yaml")
    parser.add_argument("--output_dir",  default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_history,  ppo_cfg,  ppo_time  = run_and_collect(args.ppo_config,  "PPO-Lagrangian")
    ip3o_history, ip3o_cfg, ip3o_time = run_and_collect(args.ip3o_config, "IP3O")

    cost_limit = ip3o_cfg.get("cost_limit", 25.0)

    # Save raw logs
    with open(output_dir / "ppo_history.json",  "w") as f:
        json.dump(ppo_history,  f, indent=2)
    with open(output_dir / "ip3o_history.json", "w") as f:
        json.dump(ip3o_history, f, indent=2)

    # Print summary table
    def summarize(history, label, elapsed):
        if not history:
            return
        final = history[-1]
        print(f"\n{label} ({elapsed:.0f}s)")
        print(f"  Epochs trained  : {len(history)}")
        print(f"  Final reward    : {final.get('mean_reward', float('nan')):.4f}")
        print(f"  Final cost      : {final.get('mean_cost',   float('nan')):.4f}")
        print(f"  Final KL        : {final.get('kl',          float('nan')):.5f}")
        if "penalty_coeff" in final:
            print(f"  Final penalty_coeff : {final['penalty_coeff']:.4f}")
        if "lagrange_multiplier" in final:
            print(f"  Final λ         : {final['lagrange_multiplier']:.4f}")

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    summarize(ppo_history,  "PPO-Lagrangian", ppo_time)
    summarize(ip3o_history, "IP3O",           ip3o_time)

    plot_comparison(ppo_history, ip3o_history, cost_limit, output_dir)


if __name__ == "__main__":
    main()