"""Append-only episode metrics log with console summary and JSON export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


class PerformanceTrackerTool:
    """Stores parallel lists of per-episode KPIs and can summarize or dump them to disk."""

    def __init__(self) -> None:
        # Start with empty parallel lists so log() can append rows without checking for initialization elsewhere.
        self._episodes: list[int] = []
        self._rewards: list[float] = []
        self._distances: list[float] = []
        self._promotions: list[int] = []

    def log(self, episode: int, reward: float, distance: float, promotions: int) -> None:
        # Record one completed episode’s return, travel distance, and promotion count alongside its index.
        self._episodes.append(int(episode))
        self._rewards.append(float(reward))
        self._distances.append(float(distance))
        self._promotions.append(int(promotions))

    def summary(self) -> None:
        # Print a small text table with extremes, means, and aggregate promotions so you can eyeball training health.
        if not self._episodes:
            print("(no episodes logged)")
            return
        best_i = max(range(len(self._rewards)), key=lambda i: self._rewards[i])
        worst_i = min(range(len(self._rewards)), key=lambda i: self._rewards[i])
        avg_r = sum(self._rewards) / len(self._rewards)
        avg_d = sum(self._distances) / len(self._distances)
        total_p = sum(self._promotions)
        lines = [
            "=" * 56,
            f"{'Metric':<24} {'Value':>30}",
            "-" * 56,
            f"{'Best episode (by reward)':<24} {self._episodes[best_i]:>10}  R={self._rewards[best_i]:>8.2f}  D={self._distances[best_i]:>6.1f}",
            f"{'Worst episode (by reward)':<24} {self._episodes[worst_i]:>10}  R={self._rewards[worst_i]:>8.2f}  D={self._distances[worst_i]:>6.1f}",
            f"{'Average reward':<24} {avg_r:>30.4f}",
            f"{'Average distance':<24} {avg_d:>30.4f}",
            f"{'Total promotions':<24} {total_p:>30}",
            "=" * 56,
        ]
        print("\n".join(lines))

    def export(self, path: str | Path) -> None:
        # Serialize every logged field to JSON so downstream notebooks or dashboards can load the same numbers.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": self._episodes,
            "rewards": self._rewards,
            "distances": self._distances,
            "promotions": self._promotions,
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    tr = PerformanceTrackerTool()
    tr.log(0, -100.0, 120.0, 5)
    tr.log(1, -80.0, 95.0, 8)
    tr.log(2, -90.0, 100.0, 6)
    tr.summary()
    out = Path(tempfile.gettempdir()) / "warehouse_tracker_smoke.json"
    tr.export(out)
    print("exported:", out)
