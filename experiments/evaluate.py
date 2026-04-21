"""Load training metrics and write four evaluation figures to PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    # Compute a trailing mean of the given window so the smoothed curve lines up episode-for-episode with the raw series.
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = float(a[lo : i + 1].mean())
    return out


def main() -> list[Path]:
    # Read the JSON written by train.py and build four on-disk plots comparing learning, distance, promotions, and a before/after baseline.
    base = Path(__file__).resolve().parent
    results_path = base / "results.json"
    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)

    rewards = np.asarray(data["episode_rewards"], dtype=np.float64)
    distances = np.asarray(data["episode_distances"], dtype=np.float64)
    promotions = np.asarray(data["episode_promotions"], dtype=np.float64)
    baseline_avg_distance = float(data["baseline_avg_distance"])

    episodes = np.arange(1, len(rewards) + 1)
    sns.set_theme(style="whitegrid")

    # Plot 1 — raw reward each episode plus a 50-episode rolling mean to highlight the learning trend.
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episodes, rewards, color="steelblue", alpha=0.45, linewidth=1.0, label="Episode reward")
    ax1.plot(episodes, _rolling_mean(rewards, 50), color="darkorange", linewidth=2.0, label="Rolling average (50)")
    ax1.set_title("DQN Learning Curve — Episode Reward over Training (clipped for clarity)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_ylim(-200, 50)
    ax1.legend(loc="best")
    fig1.tight_layout()
    path1 = base / "learning_curve.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # Plot 2 — distance each episode, smoothed curve, and the random-policy horizontal reference from the saved baseline scalar.
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(episodes, distances, color="steelblue", alpha=0.45, linewidth=1.0, label="Episode distance")
    ax2.plot(episodes, _rolling_mean(distances, 50), color="darkorange", linewidth=2.0, label="Rolling average (50)")
    ax2.axhline(baseline_avg_distance, color="red", linestyle="--", linewidth=2.0, label="Random baseline")
    ax2.set_title("Picker Travel Distance: Trained Agent vs Random Baseline")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Manhattan distance")
    ax2.legend(loc="best")
    fig2.tight_layout()
    path2 = base / "distance_improvement.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    # Plot 3 — for each block of ten episodes, show the mean promotion count as one bar so the chart stays readable.
    n = len(promotions)
    assert n % 10 == 0
    n_bins = n // 10
    prom_bins = promotions.reshape(n_bins, 10).mean(axis=1)
    bin_labels = [f"{i * 10 + 1}-{(i + 1) * 10}" for i in range(n_bins)]
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.barplot(x=bin_labels, y=prom_bins, ax=ax3, color="seagreen")
    ax3.set_title("Bandit Agent — Prime Zone Promotions per Episode")
    ax3.set_xlabel("Episode range (groups of 10)")
    ax3.set_ylabel("Average promotions")
    ax3.tick_params(axis="x", rotation=45)
    fig3.tight_layout()
    path3 = base / "promotion_usage.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)

    # Plot 4 — compare the saved random baseline distance to the mean distance over the last 50 training episodes and annotate the percent change.
    trained_avg_distance = float(np.mean(distances[-50:]))
    pct_improvement = (baseline_avg_distance - trained_avg_distance) / baseline_avg_distance * 100.0
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    labels = ["Random baseline", "Trained agent (avg ep 451-500)"]
    vals = [baseline_avg_distance, trained_avg_distance]
    bars = ax4.bar(labels, vals, color=["indianred", "steelblue"])
    ax4.set_title("Before vs After: Average Picker Travel Distance")
    ax4.set_ylabel("Total Manhattan distance")
    ymax = max(vals) * 1.18
    ax4.set_ylim(0.0, ymax)
    for bar, v in zip(bars, vals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax4.text(
        0.5,
        1.04,
        f"{pct_improvement:+.1f}% improvement vs baseline",
        ha="center",
        va="bottom",
        fontsize=12,
        transform=ax4.transAxes,
    )
    fig4.tight_layout()
    path4 = base / "before_after.png"
    fig4.savefig(path4, dpi=150)
    plt.close(fig4)

    paths = [path1, path2, path3, path4]
    for p in paths:
        print(p)
    return paths


if __name__ == "__main__":
    main()
