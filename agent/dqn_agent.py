from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        cfg = config["agent"]
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.target_update_freq = cfg["target_update_freq"]
        self.train_start_size = cfg["train_start_size"]

        # Epsilon-Greedy Schedule
        self.epsilon = cfg["epsilon_start"]
        self.epsilon_end = cfg["epsilon_end"]
        self.epsilon_decay_steps = cfg["epsilon_decay_steps"]
        self.epsilon_delta = (cfg["epsilon_start"] - cfg["epsilon_end"]) / cfg["epsilon_decay_steps"]

        # Dispositivo de hardware (CPU / CUDA / MPS)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Inicialização das Redes
        hidden_dim = cfg["hidden_dim"]
        self.q_policy = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_policy.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_policy.parameters(), lr=cfg["learning_rate"])
        self.replay_buffer = ReplayBuffer(cfg["buffer_capacity"])

        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_policy(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.train_start_size:
            return None

        # 1. Amostragem
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # 2. Q(s, a) atual
        current_q = self.q_policy(states).gather(1, actions)

        # 3. Target Q com Bellman Target: r + gamma * max Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.q_target(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 4. Cálculo de Loss e Backprop
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 5. Sincronização periódica da Target Network e decay de Epsilon
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_policy.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_delta)

        return float(loss.item())