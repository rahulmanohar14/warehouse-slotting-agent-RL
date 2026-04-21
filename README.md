# Warehouse Slotting RL — Multi-Agent Reinforcement Learning for Warehouse Optimization

This project simulates a 10×10 warehouse grid where twenty products with Pareto-shaped demand must be placed one at a time, combining a Deep Q-Network that chooses empty slots with a LinUCB contextual bandit that decides whether each item is promoted into a prime zone near the depot. A training script ties the environment, both learners, and logging together; evaluation scripts plot learning curves and distances; optional tools render heatmaps and track metrics; and a CrewAI orchestrator can turn saved metrics into narrative recommendations for stakeholders.

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Copy `.env.example` to `.env` and add your Groq API key (`GROQ_API_KEY`) so the CrewAI orchestrator can call the configured LLM.

## Usage

Run these commands in order:

```bash
python experiments/train.py
python experiments/evaluate.py
python tools/heatmap_tool.py
python crew/orchestrator.py
```

## Results

Training reduces total picker travel distance (sum of Manhattan distances from the depot for each placement) compared to random valid-slot assignment. In representative runs, metrics show roughly a **48.4% reduction in picker travel distance versus the random baseline**, reflecting learned preference for nearer slots under the shaped reward signal.

## Project structure

```
warehouse-slotting-rl/
├── README.md                 # This overview, setup, usage, and methodology notes
├── requirements.txt          # Pinned minimum versions for Python dependencies
├── .env.example              # Template for Groq API key used by CrewAI
├── env/
│   ├── __init__.py           # Marks env as a Python package
│   └── warehouse_env.py      # Gymnasium 10×10 slotting environment with Pareto demand and depot reward
├── agents/
│   ├── __init__.py           # Marks agents as a Python package
│   ├── dqn_agent.py          # PyTorch DQN with replay buffer, target net, and epsilon-greedy slot selection
│   └── bandit_agent.py       # NumPy LinUCB two-arm agent for promote-vs-standard zone decisions
├── crew/
│   ├── __init__.py           # Marks crew as a Python package
│   └── orchestrator.py       # CrewAI sequential crew that analyzes metrics and drafts an executive summary
├── tools/
│   ├── __init__.py           # Marks tools as a Python package
│   ├── heatmap_tool.py       # Matplotlib/seaborn heatmap export of a 100-slot demand vector
│   ├── demand_tool.py        # Builds the five-dimensional LinUCB context vector used during training
│   └── tracker_tool.py       # In-memory episode KPI log with summary table and JSON export
├── experiments/
│   ├── __init__.py           # Marks experiments as a Python package
│   ├── train.py              # 500-episode training loop wiring env, DQN, bandit, baseline, and results.json
│   ├── evaluate.py           # Loads results.json and saves four PNG evaluation plots
│   ├── results.json          # Written by train.py: per-episode rewards, distances, promotions, baseline distance
│   ├── learning_curve.png    # Episode reward and rolling mean from evaluate.py
│   ├── distance_improvement.png  # Distance series, rolling mean, and random baseline line
│   ├── promotion_usage.png   # Bar chart of average promotions per ten-episode bin
│   ├── before_after.png      # Baseline vs final-episode distance comparison with improvement text
│   └── heatmap.png           # Optional demand heatmap written by tools/heatmap_tool.py smoke or manual runs
└── notebooks/
    └── .gitkeep              # Keeps the notebooks directory in version control until notebooks are added
```

## RL methods

Deep Q-Learning (DQN) treats each empty slot as a discrete action on a 100-dimensional observation (flattened grid demands). A small multilayer perceptron approximates action values; experience replay and a slowly updated target network stabilize credit assignment while epsilon-greedy exploration ensures diverse slot visits early in training. The environment reward is the negative Manhattan distance from the depot to the chosen cell, so the policy is pushed toward placing the current product in closer, still-empty locations within whichever zone the bandit exposes.

The LinUCB contextual bandit maintains separate ridge-style design matrices for “keep in standard zone” versus “promote to prime slots (0–24)”. Each decision uses a five-feature context (demand, smoothed past returns, training progress, prior promotions for that SKU, and how tight existing placements already hug the depot). Upper-confidence scoring trades off estimated linear payoff versus uncertainty so the system explores promotion aggressively at first and gradually favors whichever arm fits the observed returns, while the DQN specializes in fine-grained slot choice inside the allowed region.
