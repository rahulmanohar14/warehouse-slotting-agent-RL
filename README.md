# Warehouse Slotting Agent RL

Multi-agent reinforcement learning system for warehouse product slotting optimization. A DQN agent learns optimal slot assignments for products in a 10x10 warehouse grid while a LinUCB contextual bandit learns dynamic prime-zone promotion decisions. Achieves a **49.4% reduction** in picker travel distance compared to a verified random baseline.

> Built for INFO 7375 — Prompt Engineering and Agentic AI Systems, Northeastern University, April 2026.

---

## What problem does this solve?

In any warehouse, workers called pickers walk from slot to slot collecting products for orders. If a frequently ordered product is stored far from the dispatch depot, the picker walks a long distance hundreds of times per day. This system learns — through reinforcement learning — to place high-demand products close to the depot, minimizing total picker travel distance.

Companies like Amazon, Walmart, Ocado, and Symbotic all operate AI-driven warehouse slotting systems at scale. This project implements a simplified but mathematically grounded version of the same core problem.

---

## Results

| Metric | Random Baseline | Trained Agent | Improvement |
|--------|----------------|---------------|-------------|
| Avg picker distance | 180.21 units | 91.18 units | **49.4% reduction** |
| Convergence episode | — | ~Episode 50 | Fast convergence |
| Bandit promotions | — | 20/20 per episode | Optimal arm found |

The agent autonomously discovered the **cube-per-order index principle** — a known optimal operations research rule — without being explicitly programmed with it.

---

## System Architecture

```
warehouse-slotting-agent-RL/
├── env/
│   └── warehouse_env.py        # Gymnasium env, 10x10 grid, Pareto demand, Manhattan reward
├── agents/
│   ├── dqn_agent.py            # DQN with replay buffer, target network, valid action masking
│   └── bandit_agent.py         # LinUCB contextual bandit, 5-feature context, 2 arms
├── crew/
│   └── orchestrator.py         # CrewAI 3-agent chain with Groq LLaMA 3.1 backend
├── tools/
│   ├── heatmap_tool.py         # Warehouse grid heatmap visualization
│   ├── demand_tool.py          # Demand forecast and context vector builder
│   └── tracker_tool.py         # KPI logging and export
|    └── visualizer.py
├── experiments/
│   ├── train.py                # 500-episode training loop, saves results.json and dqn_trained.pth
│   └── evaluate.py             # Generates 4 evaluation plots
├── notebooks/                  # Results analysis
├── requirements.txt
└── .env.example
```

---

## RL Implementation

### Method 1: Deep Q-Network (Value-Based)

- **State**: Flattened 100-dimensional grid observation (demand score per slot)
- **Action**: Slot index 0-99 (with valid action masking for empty slots only)
- **Reward**: Negative Manhattan distance from depot to chosen slot
- **Network**: Input(100) → Hidden(128, ReLU) → Hidden(128, ReLU) → Output(100)
- **Training**: Replay buffer 10,000, batch size 64, gamma 0.99, epsilon decay 1.0 → 0.05

### Method 2: LinUCB Contextual Bandit (Exploration)

- **Arms**: 0 = standard zone (slots 25-99), 1 = prime zone (slots 0-24)
- **Context**: 5 features — demand score, rolling avg demand, episode progress, promotion history, slot distance
- **Algorithm**: UCB score = theta^T * x + alpha * sqrt(x^T * A^(-1) * x), alpha = 1.0
- **Result**: Converged to always promote (arm 1) within 10 episodes

---

## Agentic Integration

Three CrewAI agents powered by Groq LLaMA 3.1 analyze training results in a sequential pipeline:

1. **Warehouse Slotting Coordinator** — analyzes the 49.4% improvement and provides 3 operational recommendations
2. **RL Performance Analyst** — interprets DQN convergence, LinUCB behavior, and exploration vs exploitation
3. **Logistics Report Writer** — produces a 150-word executive summary for a logistics director

---

## Installation

**Requirements: Python 3.11 (CrewAI does not support Python 3.14)**

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv311
venv311\Scripts\activate        # Windows
source venv311/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Setup

Copy `.env.example` to `.env` and add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

**Important on Windows**: Create the `.env` file in VS Code or use this command to ensure UTF-8 encoding:

```powershell
[System.IO.File]::WriteAllText("$PWD\.env", "GROQ_API_KEY=your_key_here`n", [System.Text.Encoding]::UTF8)
```

---

## Usage

Run in this order:

```bash
# 1. Train both RL agents for 500 episodes
python experiments/train.py

# 2. Generate evaluation plots
python experiments/evaluate.py

# 3. Generate trained agent heatmap
python tools/heatmap_tool.py

# 4. Run real-time visualizer
python tools/visualizer.py

# 5. Run CrewAI multi-agent analysis (requires GROQ_API_KEY)
# Windows PowerShell: set key first
$env:GROQ_API_KEY="your_key_here"
python crew/orchestrator.py
```

---

## Output Files

After running `train.py` and `evaluate.py`:

| File | Description |
|------|-------------|
| `experiments/results.json` | Episode rewards, distances, promotions, baseline |
| `experiments/dqn_trained.pth` | Saved DQN policy network weights |
| `experiments/learning_curve.png` | DQN reward over 500 episodes |
| `experiments/distance_improvement.png` | Trained agent vs random baseline distance |
| `experiments/before_after.png` | Before/after bar chart with % improvement |
| `experiments/promotion_usage.png` | LinUCB bandit arm selection per episode |
| `experiments/heatmap_trained.png` | Final warehouse grid state heatmap |

---

## Tech Stack

| Package | Version | Role |
|---------|---------|------|
| Python | 3.11.9 | Runtime |
| PyTorch | 2.11.0 | DQN neural network |
| Gymnasium | 1.2.3 | RL environment |
| NumPy | 2.4.4 | LinUCB matrix operations |
| CrewAI | 1.14.2 | Multi-agent orchestration |
| LiteLLM | 1.83.10 | Groq API abstraction |
| Matplotlib / Seaborn | 3.10 / 0.13 | Visualization |

---

## Real-World Relevance

This project addresses the warehouse slotting problem — the same core problem solved by:

- **Amazon** — Kiva robot fulfillment centers
- **Ocado** — automated grocery warehouse slotting
- **Symbotic** — RL-powered slotting for Walmart distribution
- **Alert Innovation** — acquired by Walmart for warehouse AI

---

## Author

Rahul Manohar Durshi
MS Information Systems, Northeastern University
[GitHub](https://github.com/rahulmanohar14)