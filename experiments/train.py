"""Train DQN slot placement with LinUCB promotion gating; log metrics and save results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.warehouse_env import WarehouseEnv
from agents.dqn_agent import DQNAgent
from agents.bandit_agent import LinUCBAgent


def _manhattan(slot: int, grid_size: int = 10) -> int:
    # Convert a flat slot index into row and column, then return the depot Manhattan distance used by the environment reward.
    row, col = divmod(slot, grid_size)
    return int(row + col)


def _best_occupied_distance(env: WarehouseEnv) -> float:
    # Among slots that already hold a product, find the smallest depot distance; if the grid is still empty, return the worst-case distance 18.
    assert env._grid is not None
    occupied = np.flatnonzero(env._grid > 0)
    if occupied.size == 0:
        return 18.0
    return float(min(_manhattan(int(s)) for s in occupied))


def _rolling_reward_normalized(episode_rewards: list[float], current_ep: int) -> float:
    # Average total return over up to the five most recently finished episodes, then squash it into a 0–1 range using the min/max of all finished episodes so far.
    if current_ep <= 0 or not episode_rewards:
        return 0.5
    start = max(0, current_ep - 5)
    window = episode_rewards[start:current_ep]
    raw = float(np.mean(window)) if window else 0.0
    hist = episode_rewards[:current_ep]
    if len(hist) < 2:
        return 0.5
    rmin, rmax = float(min(hist)), float(max(hist))
    span = rmax - rmin
    if span < 1e-8:
        return 0.5
    return float(np.clip((raw - rmin) / span, 0.0, 1.0))


def run_training() -> tuple[list[float], list[float], list[int]]:
    # Run 500 episodes where LinUCB chooses a prime vs standard zone, DQN picks an empty slot inside that zone, and both agents learn from the shaped step reward.
    env = WarehouseEnv()
    dqn = DQNAgent()
    bandit = LinUCBAgent()

    # Collect one scalar summary per finished episode so we can log curves and build bandit context from recent returns.
    episode_rewards: list[float] = []
    episode_distances: list[float] = []
    episode_promotions: list[int] = []
    promo_by_product = np.zeros(env.n_products, dtype=np.int64)

    for ep in range(500):
        # Begin a fresh warehouse layout and zero out the per-episode counters before placing twenty products.
        obs, _ = env.reset()
        ep_return = 0.0
        ep_distance = 0.0
        ep_promos = 0

        # Sequentially place every product: build context, ask the bandit for a zone, ask the DQN for a slot, then update both learners.
        for _step in range(20):
            assert env._demands is not None and env._grid is not None
            product_idx = int(env._product_idx)
            demand = float(env._demands[product_idx])

            # Build the five-dimensional context vector shared with the bandit before it chooses promotion.
            feat_roll = _rolling_reward_normalized(episode_rewards, ep)
            context = np.array(
                [
                    demand,
                    feat_roll,
                    ep / 500.0,
                    promo_by_product[product_idx] / 10.0,
                    _best_occupied_distance(env) / 18.0,
                ],
                dtype=np.float64,
            )

            bandit_action = bandit.select_action(context)
            promotion_bonus = bandit.get_promotion_bonus(bandit_action, demand)
            if bandit_action == 1:
                ep_promos += 1
                promo_by_product[product_idx] += 1

            # Restrict DQN to empty prime slots when promoted, otherwise to empty standard slots.
            if bandit_action == 1:
                zone = range(0, 25)
            else:
                zone = range(25, 100)
            valid_slots = [s for s in zone if env._grid[s] == 0.0]
            if not valid_slots:
                valid_slots = [s for s in range(env.n_slots) if env._grid[s] == 0.0]

            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            slot = dqn.select_action(obs_t, valid_slots)
            next_obs, env_reward, done, _trunc, _info = env.step(slot)
            dist = _manhattan(slot)
            ep_distance += dist

            total_reward = float(env_reward) + float(promotion_bonus)
            ep_return += total_reward

            next_t = torch.as_tensor(next_obs, dtype=torch.float32)
            dqn.store(obs_t, slot, total_reward, next_t, done)
            dqn.train_step()
            bandit.update(bandit_action, context, total_reward)

            obs = next_obs

        episode_rewards.append(ep_return)
        episode_distances.append(ep_distance)
        episode_promotions.append(ep_promos)

        # Emit a short progress line every 50 episodes using simple averages over the latest window.
        if (ep + 1) % 50 == 0:
            w = slice(ep + 1 - 50, ep + 1)
            avg_r = float(np.mean(episode_rewards[w]))
            avg_d = float(np.mean(episode_distances[w]))
            avg_p = float(np.mean(episode_promotions[w]))
            print(
                f"Episode {ep + 1} | Avg Reward: {avg_r:.4f} | Avg Distance: {avg_d:.4f} | Promotions: {avg_p:.1f}"
            )

    torch.save(dqn.policy_net.state_dict(), "experiments/dqn_trained.pth")
    print("Saved trained model to experiments/dqn_trained.pth")

    return episode_rewards, episode_distances, episode_promotions


def run_baseline(env: WarehouseEnv, n_episodes: int = 100, seed: int = 12345) -> float:
    # Measure how far random valid placements travel on average so we can compare learned behavior against an unguided policy.
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    for _ in range(n_episodes):
        _obs, _ = env.reset()
        dist_sum = 0.0
        for _ in range(20):
            empty = [i for i in range(env.n_slots) if env._grid[i] == 0.0]
            slot = int(rng.choice(empty))
            dist_sum += _manhattan(slot)
            env.step(slot)
        totals.append(dist_sum)
    return float(np.mean(totals))


def save_results(
    path: Path,
    episode_rewards: list[float],
    episode_distances: list[float],
    episode_promotions: list[int],
    baseline_avg_distance: float,
) -> None:
    # Persist per-episode training stats plus the random-placement baseline distance into a single JSON file for plotting later.
    payload = {
        "episode_rewards": episode_rewards,
        "episode_distances": episode_distances,
        "episode_promotions": episode_promotions,
        "baseline_avg_distance": baseline_avg_distance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    # Execute the full experiment pipeline: train agents, evaluate a random baseline, and write all scalars to disk.
    rewards, distances, promotions = run_training()
    baseline_env = WarehouseEnv()
    baseline_avg = run_baseline(baseline_env)
    out_path = Path(__file__).resolve().parent / "results.json"
    save_results(out_path, rewards, distances, promotions, baseline_avg)
    print(f"Saved results to {out_path}")
