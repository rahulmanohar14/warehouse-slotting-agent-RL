"""LinUCB contextual bandit for promote-vs-standard slotting decisions (NumPy only)."""

from __future__ import annotations

import numpy as np


class LinUCBAgent:
    """Two-arm LinUCB: arm 0 keeps a product in the standard zone, arm 1 promotes it into the prime zone (slots 0–24)."""

    def __init__(self, n_features: int = 5, alpha: float = 1.0) -> None:
        # Allocate one ridge-style design matrix and reward-weight vector per arm, plus the exploration strength alpha.
        self.n_features = n_features
        self.alpha = alpha
        self.n_arms = 2
        self._A = [np.eye(n_features, dtype=np.float64) for _ in range(self.n_arms)]
        self._b = [np.zeros(n_features, dtype=np.float64) for _ in range(self.n_arms)]

    def select_action(self, context: np.ndarray) -> int:
        # Score each arm with estimated mean reward plus an uncertainty bonus, then return the arm with the higher LinUCB score.
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"context must have length {self.n_features}, got {x.shape[0]}")

        scores = np.zeros(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            A = self._A[a]
            b = self._b[a]
            theta = np.linalg.solve(A, b)
            mean = float(theta @ x)
            inv_A_x = np.linalg.solve(A, x)
            bonus = self.alpha * float(np.sqrt(max(0.0, x @ inv_A_x)))
            scores[a] = mean + bonus

        return int(np.argmax(scores))

    def update(self, action: int, context: np.ndarray, reward: float) -> None:
        # After observing a reward for the chosen arm, add the outer product of the context to that arm’s A and add reward-weighted context to b.
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"context must have length {self.n_features}, got {x.shape[0]}")

        self._A[action] = self._A[action] + np.outer(x, x)
        self._b[action] = self._b[action] + float(reward) * x

    def get_promotion_bonus(self, action: int, demand_score: float) -> float:
        # Return an extra reward shaped like business value from promotion: half the demand score when promoting, otherwise zero.
        if int(action) == 1:
            return 0.5 * float(demand_score)
        return 0.0


if __name__ == "__main__":
    agent = LinUCBAgent()
    counts = {0: 0, 1: 0}
    rng = np.random.default_rng(0)
    for _ in range(10):
        context = rng.standard_normal(5)
        arm = agent.select_action(context)
        counts[arm] += 1
        agent.update(arm, context, float(rng.standard_normal()))
    print("arm 0 count:", counts[0], "arm 1 count:", counts[1])
