"""Live matplotlib animation of DQN training on the warehouse slotting environment."""

from __future__ import annotations

import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from agents.dqn_agent import DQNAgent
from env.warehouse_env import WarehouseEnv


class WarehouseVisualizer:
    """Animates a heatmap and KPI sidebar while a DQN trains on WarehouseEnv in lockstep with each frame."""

    def __init__(self) -> None:
        # Hold references to the matplotlib animation and axes so they stay alive for the duration of the show() loop.
        self._anim: animation.FuncAnimation | None = None
        self._fig: plt.Figure | None = None
        self._env: WarehouseEnv | None = None
        self._dqn: DQNAgent | None = None
        self._n_episodes: int = 0
        self._baseline: float = 180.21
        self._obs: np.ndarray | None = None
        self._ep_reward: float = 0.0
        self._ep_distance: float = 0.0
        self._cum_distance_all_eps: float = 0.0
        self._im = None
        self._ax_heat = None
        self._ax_side = None
        self._ax_prog = None

    def _update_frame(self, frame: int) -> None:
        # Perform one real placement and learning update, then refresh the heatmap, title, sidebar text, and bottom progress bar.
        assert self._env is not None and self._dqn is not None and self._obs is not None
        assert self._ax_heat is not None and self._ax_side is not None and self._ax_prog is not None
        assert self._im is not None

        ep = frame // 20
        sub = frame % 20
        if sub == 0:
            self._obs, _ = self._env.reset()
            self._ep_reward = 0.0
            self._ep_distance = 0.0

        assert self._env._grid is not None
        valid_slots = [i for i in range(self._env.n_slots) if self._env._grid[i] == 0.0]
        obs_t = torch.as_tensor(self._obs, dtype=torch.float32)
        slot = self._dqn.select_action(obs_t, valid_slots)
        next_obs, reward, done, _trunc, _info = self._env.step(slot)
        row, col = divmod(slot, self._env.grid_size)
        self._ep_distance += float(row + col)
        self._cum_distance_all_eps += float(row + col)
        self._ep_reward += float(reward)

        next_t = torch.as_tensor(next_obs, dtype=torch.float32)
        self._dqn.store(obs_t, slot, float(reward), next_t, done)
        self._dqn.train_step()

        self._obs = next_obs

        grid = self._env._grid.reshape(self._env.grid_size, self._env.grid_size)
        self._im.set_data(grid)
        vmax = max(float(np.max(grid)), 1e-6)
        self._im.set_clim(0.0, vmax)

        self._ax_heat.set_title(
            f"Episode {ep + 1}/{self._n_episodes} | "
            f"Return so far: {self._ep_reward:.2f} | "
            f"Last step reward: {float(reward):.2f} | "
            f"Distance so far: {self._ep_distance:.0f}"
        )

        pct_ep = (self._baseline - self._ep_distance) / self._baseline * 100.0 if self._baseline > 0 else 0.0
        self._ax_side.clear()
        self._ax_side.axis("off")
        lines = [
            "Live stats",
            f"Current episode: {ep + 1} / {self._n_episodes}",
            f"Total distance (all eps): {self._cum_distance_all_eps:.1f}",
            f"Distance this episode: {self._ep_distance:.1f}",
            f"Baseline (random / ep): {self._baseline:.2f}",
            f"% improvement vs baseline: {pct_ep:+.1f}%",
        ]
        y0 = 0.92
        for i, line in enumerate(lines):
            self._ax_side.text(0.05, y0 - i * 0.14, line, transform=self._ax_side.transAxes, fontsize=10)

        self._ax_prog.clear()
        progress = (ep + (sub + 1) / 20.0) / float(self._n_episodes)
        self._ax_prog.barh([0], [progress], height=0.55, color="steelblue", align="center")
        self._ax_prog.set_xlim(0.0, 1.0)
        self._ax_prog.set_yticks([])
        self._ax_prog.set_xlabel(f"Training progress — episode {ep + 1} of {self._n_episodes}")
        self._ax_prog.set_title("Episodes completed (fraction of total)")

    def animate_training(self, n_episodes: int, speed_ms: int = 200) -> None:
        # Build a fresh environment and DQN, lay out the figure with heatmap, stats panel, and progress row, then run FuncAnimation for every placement.
        self._env = WarehouseEnv()
        self._dqn = DQNAgent()
        self._n_episodes = int(n_episodes)
        self._baseline = 180.21
        self._cum_distance_all_eps = 0.0
        self._obs, _ = self._env.reset()
        self._ep_reward = 0.0
        self._ep_distance = 0.0

        self._fig = plt.figure(figsize=(11, 7))
        gs = self._fig.add_gridspec(2, 2, height_ratios=[1, 0.22], width_ratios=[3, 1], hspace=0.35, wspace=0.25)
        self._ax_heat = self._fig.add_subplot(gs[0, 0])
        self._ax_side = self._fig.add_subplot(gs[0, 1])
        self._ax_prog = self._fig.add_subplot(gs[1, :])

        grid0 = np.zeros((self._env.grid_size, self._env.grid_size), dtype=np.float64)
        self._im = self._ax_heat.imshow(
            grid0,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        self._fig.colorbar(self._im, ax=self._ax_heat, fraction=0.046, pad=0.04, label="Demand")
        self._ax_heat.set_xlabel("Column")
        self._ax_heat.set_ylabel("Row")

        total_frames = self._n_episodes * 20
        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            frames=range(total_frames),
            interval=int(speed_ms),
            repeat=False,
            blit=False,
        )
        plt.show()


if __name__ == "__main__":
    WarehouseVisualizer().animate_training(n_episodes=50, speed_ms=150)
