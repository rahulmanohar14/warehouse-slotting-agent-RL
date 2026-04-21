"""Gymnasium environment for sequential warehouse slotting on a 10x10 grid."""

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces


class WarehouseEnv(Env):
    """Place 20 Pareto-distributed products into a 10x10 grid; reward favors slots near the depot at (0,0)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 10,
        n_products: int = 20,
        pareto_shape: float = 1.5,
    ) -> None:
        # Set up grid dimensions, product count, and the Pareto shape that controls how spiky demand is (smaller = stronger 80/20 tail).
        self.grid_size = grid_size
        self.n_slots = grid_size * grid_size
        self.n_products = n_products
        self.pareto_shape = pareto_shape

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.n_slots,),
            dtype=np.float64,
        )
        self.action_space = spaces.Discrete(self.n_slots)

        self._demands: np.ndarray | None = None
        self._grid: np.ndarray | None = None
        self._product_idx: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        # Start a new episode: optionally re-seed, sample fresh product demands, clear the grid, and point at the first product to place.
        super().reset(seed=seed)

        self._demands = self.np_random.pareto(self.pareto_shape, size=self.n_products)
        self._grid = np.zeros(self.n_slots, dtype=np.float64)
        self._product_idx = 0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Apply one placement decision: validate the slot, optionally write the current product demand, compute reward as negative depot distance, and advance or end the episode when all products are placed.
        assert self._grid is not None and self._demands is not None

        info: dict = {"invalid": False}

        if not (0 <= action < self.n_slots):
            info["invalid"] = True
            return self._get_obs(), -float(self.n_slots), False, False, info

        if self._grid[action] > 0.0:
            info["invalid"] = True
            return self._get_obs(), -float(self.n_slots), False, False, info

        if self._product_idx >= self.n_products:
            info["invalid"] = True
            return self._get_obs(), 0.0, True, False, info

        row, col = divmod(action, self.grid_size)
        manhattan = float(row + col)
        reward = -manhattan

        self._grid[action] = self._demands[self._product_idx]
        self._product_idx += 1

        terminated = self._product_idx >= self.n_products
        return self._get_obs(), reward, terminated, False, info

    def render(self) -> None:
        # Print a human-readable 10x10 ASCII view of each slot's stored demand (0 means empty).
        if self._grid is None:
            print("(call reset() before render())")
            return

        g = self._grid.reshape(self.grid_size, self.grid_size)
        for r in range(self.grid_size):
            row_strs = [f"{g[r, c]:>8.3f}" for c in range(self.grid_size)]
            print(" ".join(row_strs))
        print()

    def _get_obs(self) -> np.ndarray:
        # Return the current observation vector (flattened slot demands, zeros for empty slots).
        assert self._grid is not None
        return self._grid.astype(np.float64, copy=True)
