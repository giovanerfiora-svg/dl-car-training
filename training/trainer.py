from typing import Any, Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from environment.vehicle_env import VehicleEnv
from agent.dqn_agent import DQNAgent
from training.checkpoint import CheckpointManager


class Trainer:
    """Orquestra o treinamento e registra métricas para análise no TensorBoard."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = VehicleEnv(config)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = DQNAgent(state_dim, action_dim, config)

        training_cfg = config["training"]
        self.ckpt_manager = CheckpointManager(training_cfg["save_dir"])
        self.writer = SummaryWriter(log_dir=training_cfg["log_dir"])
        self.total_timesteps = int(training_cfg["total_timesteps"])
        self.checkpoint_freq = int(training_cfg["checkpoint_freq_steps"])
        self.metric_log_freq = int(training_cfg.get("metric_log_freq_steps", 100))

    def _log_step_metrics(self, loss: float | None) -> None:
        if self.agent.total_steps == 0 or self.agent.total_steps % self.metric_log_freq != 0:
            return
        self.writer.add_scalar("Train/Loss", loss or 0.0, self.agent.total_steps)
        self.writer.add_scalar("Train/Epsilon", self.agent.epsilon, self.agent.total_steps)
        self.writer.add_scalar("Train/ReplayBufferSize", len(self.agent.replay_buffer), self.agent.total_steps)
        self.writer.add_scalar("Train/LearningRate", self.agent.optimizer.param_groups[0]["lr"], self.agent.total_steps)
        self.writer.add_scalar("Train/GradientNorm", getattr(self.agent, "last_grad_norm", 0.0), self.agent.total_steps)

    def train(self) -> None:
        start_episode, _, best_reward, loaded = self.ckpt_manager.load_latest(self.agent)
        if loaded:
            print(f"[Trainer] Retomando treino do episódio {start_episode} ({self.agent.total_steps} passos).")
        else:
            print("[Trainer] Iniciando novo treinamento do zero.")

        episode = start_episode
        recent_rewards: list[float] = []
        pbar = tqdm(total=self.total_timesteps, initial=self.agent.total_steps, desc="Treinando Agente")

        try:
            while self.agent.total_steps < self.total_timesteps:
                episode += 1
                obs, info = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_collisions = 0
                episode_checkpoints = 0
                speed_sum = 0.0
                losses: list[float] = []
                done = False

                while not done and self.agent.total_steps < self.total_timesteps:
                    action = self.agent.select_action(obs, evaluate=False)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    self.agent.replay_buffer.push(obs, action, reward, next_obs, done)
                    obs = next_obs

                    episode_reward += float(reward)
                    episode_steps += 1
                    speed_sum += float(info.get("speed", 0.0))
                    episode_collisions += int(info.get("is_crashed", False))
                    episode_checkpoints += int(info.get("reached_checkpoint", False))

                    loss = self.agent.update()
                    if loss is not None:
                        losses.append(loss)
                    pbar.update(1)
                    self._log_step_metrics(loss)

                    if self.agent.total_steps % self.checkpoint_freq == 0 and self.agent.total_steps > 0:
                        avg_rw = float(np.mean(recent_rewards[-50:])) if recent_rewards else episode_reward
                        is_best = avg_rw > best_reward
                        if is_best:
                            best_reward = avg_rw
                        self.ckpt_manager.save(self.agent, episode, self.agent.total_steps, best_reward, avg_rw, is_best=is_best)

                recent_rewards.append(episode_reward)
                avg_50 = float(np.mean(recent_rewards[-50:]))
                mean_speed = speed_sum / max(episode_steps, 1)
                mean_loss = float(np.mean(losses)) if losses else 0.0

                self.writer.add_scalar("Episode/Reward", episode_reward, episode)
                self.writer.add_scalar("Episode/AverageReward50", avg_50, episode)
                self.writer.add_scalar("Episode/Steps", episode_steps, episode)
                self.writer.add_scalar("Episode/AverageSpeed", mean_speed, episode)
                self.writer.add_scalar("Episode/Checkpoints", episode_checkpoints, episode)
                self.writer.add_scalar("Episode/Collisions", episode_collisions, episode)
                self.writer.add_scalar("Episode/MeanLoss", mean_loss, episode)
                self.writer.add_scalar("Episode/CompletionRatio", episode_steps / max(self.env.max_steps, 1), episode)
                pbar.set_postfix(reward=f"{episode_reward:.2f}", avg50=f"{avg_50:.2f}", eps=f"{self.agent.epsilon:.3f}", loss=f"{mean_loss:.4f}")
        finally:
            pbar.close()
            self.writer.close()

        print("[Trainer] Treinamento concluído com sucesso!")
