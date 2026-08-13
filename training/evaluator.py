import numpy as np
from typing import Dict, Any
from environment.vehicle_env import VehicleEnv
from agent.dqn_agent import DQNAgent


class Evaluator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = VehicleEnv(config)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = DQNAgent(state_dim, action_dim, config)

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        rewards = []
        crashes = 0
        steps_list = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            steps = 0
            done = False

            while not done:
                # evaluate=True força o agente a usar 100% da rede aprendida (Epsilon = 0)
                action = self.agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1

            rewards.append(episode_reward)
            steps_list.append(steps)
            if info.get("is_crashed", False):
                crashes += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "crash_rate": float(crashes / num_episodes),
            "mean_steps": float(np.mean(steps_list)),
        }