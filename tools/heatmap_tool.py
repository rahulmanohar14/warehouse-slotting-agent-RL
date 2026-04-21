"""Render warehouse demand grids as heatmaps for quick visual inspection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from agents.dqn_agent import DQNAgent
from env.warehouse_env import WarehouseEnv


class HeatmapTool:
    """Writes a 10x10 demand heatmap image with the depot cell highlighted."""

    def render(self, grid_state: np.ndarray, title: str, output_path: Path | None = None) -> Path:
        # Turn the flat 100-length state into a grid, draw a yellow–red heatmap with a boxed depot label, save PNG to the default or chosen path, and return where it was written.
        data = np.asarray(grid_state, dtype=np.float64).reshape(10, 10)
        if output_path is None:
            out_path = Path(__file__).resolve().parents[1] / "experiments" / "heatmap.png"
        else:
            out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            data,
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Demand"},
            linewidths=0.5,
            linecolor="lightgray",
        )
        ymin, ymax = ax.get_ylim()
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        depot_y0 = ymin
        rect = patches.Rectangle(
            (0, depot_y0),
            1,
            1,
            linewidth=2.5,
            edgecolor="black",
            facecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            depot_y0 + 0.5,
            "Depot",
            ha="center",
            va="center",
            color="black",
            fontsize=11,
            fontweight="bold",
            zorder=11,
        )
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def render_from_results(self) -> Path:
        # Run five full DQN episodes (epsilon=0.05 when weights exist), sum each episode's final slot-demand grid, average over episodes, then save the heatmap.
        env = WarehouseEnv()
        dqn = DQNAgent()
        weights_path = Path(__file__).parent.parent / "experiments" / "dqn_trained.pth"
        if weights_path.exists():
            dqn.policy_net.load_state_dict(torch.load(weights_path, map_location="cpu"))
            dqn.epsilon = 0.05

        n_episodes = 5
        slot_demand_sum = np.zeros(env.n_slots, dtype=np.float64)

        for _ in range(n_episodes):
            obs, _ = env.reset()
            terminated = False
            while not terminated:
                assert env._grid is not None
                valid_slots = [i for i in range(env.n_slots) if env._grid[i] == 0.0]
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                slot = dqn.select_action(obs_t, valid_slots)
                obs, _reward, terminated, _trunc, _info = env.step(slot)
            slot_demand_sum += np.asarray(obs, dtype=np.float64)

        avg_grid = slot_demand_sum / float(n_episodes)
        trained_path = Path(__file__).resolve().parents[1] / "experiments" / "heatmap_trained.png"
        return self.render(
            avg_grid,
            title="Trained Agent — Final Warehouse Grid State",
            output_path=trained_path,
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tool = HeatmapTool()
    grid = rng.random(100) * 5.0
    path = tool.render(grid, "Smoke test heatmap")
    print(path)
    path_trained = tool.render_from_results()
    print(path_trained)
