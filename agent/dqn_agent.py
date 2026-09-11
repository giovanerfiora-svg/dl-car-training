from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """Agente Double DQN com arquitetura Dueling opcional."""

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        cfg = config["agent"]
        self.gamma = float(cfg["gamma"])
        self.batch_size = int(cfg["batch_size"])
        self.target_update_freq = int(cfg["target_update_freq"])
        self.train_start_size = int(cfg["train_start_size"])
        self.dueling = bool(cfg.get("dueling", True))

        self.epsilon = float(cfg["epsilon_start"])
        self.epsilon_end = float(cfg["epsilon_end"])
        self.epsilon_decay_steps = int(cfg["epsilon_decay_steps"])
        self.epsilon_delta = (
            self.epsilon - self.epsilon_end
        ) / max(self.epsilon_decay_steps, 1)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        hidden_dim = int(cfg["hidden_dim"])
        self.q_policy = QNetwork(
            state_dim, action_dim, hidden_dim, dueling=self.dueling
        ).to(self.device)
        self.q_target = QNetwork(
            state_dim, action_dim, hidden_dim, dueling=self.dueling
        ).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(
            self.q_policy.parameters(), lr=float(cfg["learning_rate"])
        )
        self.replay_buffer = ReplayBuffer(int(cfg["buffer_capacity"]))
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q_policy(state_t), dim=1).item())

    def update(self) -> float | None:
        if len(self.replay_buffer) < max(self.train_start_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        current_q = self.q_policy(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.q_target(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_policy.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon = max(
                self.epsilon_end, self.epsilon - self.epsilon_delta
            )

        self.last_grad_norm = float(grad_norm)
        return float(loss.item())
