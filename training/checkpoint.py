import os
from typing import Dict, Any, Tuple
import torch

from agent.dqn_agent import DQNAgent


class CheckpointManager:

    def __init__(self, save_dir: str = "models"):
        self.save_dir = save_dir
        self.checkpoints_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.latest_path = os.path.join(self.save_dir, "latest.pt")
        self.best_path = os.path.join(self.save_dir, "best.pt")

    def save(
        self,
        agent: DQNAgent,
        episode: int,
        total_steps: int,
        best_reward: float,
        recent_avg_reward: float,
        is_best: bool = False,
    ) -> None:
        checkpoint = {
            "q_policy_state_dict": agent.q_policy.state_dict(),
            "q_target_state_dict": agent.q_target.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "total_steps": agent.total_steps,
            "episode": episode,
            "best_reward": best_reward,
            "recent_avg_reward": recent_avg_reward,
        }

        # 1. Salva o estado mais recente
        torch.save(checkpoint, self.latest_path)

        # 2. Salva uma cópia se for o melhor modelo até agora
        if is_best:
            torch.save(checkpoint, self.best_path)

        # 3. Save periódico de histórico na subpasta checkpoints
        step_path = os.path.join(
            self.checkpoints_dir, f"checkpoint_step_{agent.total_steps}.pt"
        )
        torch.save(checkpoint, step_path)

    def load_latest(
        self, agent: DQNAgent
    ) -> Tuple[int, int, float, bool]:
        
        if not os.path.exists(self.latest_path):
            return 0, 0, -float("inf"), False

        print(f"[Checkpoint] Carregando checkpoint existente: {self.latest_path}")
        checkpoint = torch.load(self.latest_path, map_location=agent.device)

        agent.q_policy.load_state_dict(checkpoint["q_policy_state_dict"])
        agent.q_target.load_state_dict(checkpoint["q_target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        agent.total_steps = checkpoint["total_steps"]

        episode = checkpoint["episode"]
        best_reward = checkpoint["best_reward"]

        return episode, agent.total_steps, best_reward, True