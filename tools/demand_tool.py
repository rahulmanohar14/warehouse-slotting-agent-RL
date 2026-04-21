"""Build the same five-dimensional bandit context vector used during RL training."""

from __future__ import annotations

import numpy as np


class DemandForecastTool:
    """Maps raw telemetry into a fixed-length context array for LinUCB-style decisions."""

    @staticmethod
    def rolling_avg(history: np.ndarray | list[float], window: int = 5) -> float:
        # Return the simple mean of the last `window` scalar entries (or all entries if the run is shorter than the window).
        h = np.asarray(history, dtype=np.float64).ravel()
        if h.size == 0:
            return 0.0
        k = min(int(window), int(h.size))
        return float(h[-k:].mean())

    def get_context(
        self,
        demand_score: float,
        reward_history: np.ndarray | list[float],
        episode: int,
        promo_count: int,
        slot_distance: float,
    ) -> np.ndarray:
        # Stack demand, normalized rolling episode return, progress, promotion intensity, and best-slot distance exactly like experiments/train.py.
        rh = [float(x) for x in np.asarray(reward_history, dtype=np.float64).ravel()]
        current_ep = int(episode)
        if current_ep <= 0 or len(rh) == 0:
            feat_roll = 0.5
        else:
            start = max(0, current_ep - 5)
            window = rh[start:current_ep]
            raw = float(np.mean(window)) if window else 0.0
            hist = rh[:current_ep]
            if len(hist) < 2:
                feat_roll = 0.5
            else:
                rmin, rmax = float(min(hist)), float(max(hist))
                span = rmax - rmin
                if span < 1e-8:
                    feat_roll = 0.5
                else:
                    feat_roll = float(np.clip((raw - rmin) / span, 0.0, 1.0))

        return np.array(
            [
                float(demand_score),
                feat_roll,
                current_ep / 500.0,
                float(promo_count) / 10.0,
                float(slot_distance) / 18.0,
            ],
            dtype=np.float64,
        )


if __name__ == "__main__":
    tool = DemandForecastTool()
    hist = [-80.0, -75.0, -70.0, -72.0, -68.0, -65.0]
    ctx = tool.get_context(2.5, hist, episode=6, promo_count=2, slot_distance=9.0)
    ra = tool.rolling_avg(hist, window=5)
    print("rolling_avg:", ra)
    print("context:", ctx)
