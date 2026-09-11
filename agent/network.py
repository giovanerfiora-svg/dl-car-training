import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Dueling Q-Network: separa valor do estado e vantagem das ações."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if dueling:
            self.value_head = nn.Linear(hidden_dim, 1)
            self.advantage_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        if not self.dueling:
            return self.out(x)

        value = self.value_head(x)
        advantage = self.advantage_head(x)

        # Q(s,a) = V(s) + A(s,a) - média[A(s,a)]
        return value + advantage - advantage.mean(dim=1, keepdim=True)
