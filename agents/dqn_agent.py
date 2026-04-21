"""Deep Q-Network agent for discrete slot placement (PyTorch only)."""

from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Feedforward network mapping a 100-dim state to Q-values for 100 slot actions."""

    def __init__(self) -> None:
        # Build two 128-neuron hidden layers with ReLU activations and a linear output head of size 100.
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run a batch of state vectors through the MLP and return one Q-value per slot for each row in the batch.
        return self.net(x)


class DQNAgent:
    """DQN with experience replay, epsilon-greedy exploration, and a periodically refreshed target network."""

    def __init__(
        self,
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        target_update_interval: int = 100,
        device: torch.device | None = None,
    ) -> None:
        # Wire up policy and target networks on the chosen device, create the Adam optimizer, and initialize replay, epsilon, and step counters.
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=replay_capacity)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self._learn_steps = 0

        self.update_target()

    def select_action(self, obs: torch.Tensor, valid_slots: list[int]) -> int:
        # With probability epsilon pick a random empty slot; otherwise pick the empty slot with the highest predicted Q-value from the policy network.
        if not valid_slots:
            raise ValueError("valid_slots must be non-empty")

        if random.random() < self.epsilon:
            return random.choice(valid_slots)

        self.policy_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            q = self.policy_net(x).squeeze(0)
            mask = torch.full((100,), -float("inf"), device=self.device)
            mask[valid_slots] = q[valid_slots]
            return int(torch.argmax(mask).item())

    def store(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        # Push one transition into the replay buffer (as CPU tensors) and, if the transition ends an episode, decay epsilon toward its minimum floor.
        self.replay.append(
            (
                state.detach().cpu().float().view(-1),
                int(action),
                float(reward),
                next_state.detach().cpu().float().view(-1),
                bool(done),
            )
        )
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self) -> None:
        # If enough samples exist, draw a minibatch, apply one step of DQN TD learning on the policy net, and refresh the target net on a fixed schedule.
        if len(self.replay) < self.batch_size:
            return

        self.policy_net.train()
        batch = random.sample(self.replay, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([b[3] for b in batch]).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_interval == 0:
            self.update_target()

    def update_target(self) -> None:
        # Copy all policy-network weights into the target network so bootstrapped targets stay stable between updates.
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    agent = DQNAgent()
    obs = torch.randn(100)
    valid_slots = list(range(100))
    action = agent.select_action(obs, valid_slots)
    print(action)
