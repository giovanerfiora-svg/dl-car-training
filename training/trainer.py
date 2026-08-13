import os
from typing import Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from environment.vehicle_env import VehicleEnv
from agent.dqn_agent import DQNAgent
from training.checkpoint import CheckpointManager


class Trainer:
    """Orquestrador do treinamento contínuo em segundo plano."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = VehicleEnv(config)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.agent = DQNAgent(state_dim, action_dim, config)
        self.ckpt_manager = CheckpointManager(config["training"]["save_dir"])

        # TensorBoard Logger
        log_dir = config["training"]["log_dir"]
        self.writer = SummaryWriter(log_dir=log_dir)

        self.total_timesteps = config["training"]["total_timesteps"]
        self.checkpoint_freq = config["training"]["checkpoint_freq_steps"]

    def train(self) -> None:
        """Executa o loop principal de treinamento com suporte a resumo automático."""
        start_episode, start_steps, best_reward, loaded = (
            self.ckpt_manager.load_latest(self.agent)
        )

        if loaded:
            print(f"[Trainer] Retomando treino do episódio {start_episode} ({start_steps} passos).")
        else:
            print("[Trainer] Iniciando novo treinamento do zero.")

        episode = start_episode
        recent_rewards = []
        pbar = tqdm(total=self.total_timesteps, initial=self.agent.total_steps, desc="Treinando Agente")

        while self.agent.total_steps < self.total_timesteps:
            episode += 1
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                # 1. Selecionar e aplicar ação
                action = self.agent.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 2. Armazenar no Replay Buffer
                self.agent.replay_buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs

                episode_reward += reward
                episode_steps += 1

                # 3. Atualizar a Rede Neural
                loss = self.agent.update()

                # Update na barra de progresso TQDM
                pbar.update(1)

                # 4. Registrar métricas por passo no TensorBoard
                if loss is not None and self.agent.total_steps % 100 == 0:
                    self.writer.add_scalar("Train/Loss", loss, self.agent.total_steps)
                    self.writer.add_scalar("Train/Epsilon", self.agent.epsilon, self.agent.total_steps)

                # 5. Checkpoint periódico por quantidade de passos
                if self.agent.total_steps % self.checkpoint_freq == 0 and self.agent.total_steps > 0:
                    avg_rw = float(np.mean(recent_rewards[-50:])) if recent_rewards else episode_reward
                    is_best = avg_rw > best_reward
                    if is_best:
                        best_reward = avg_rw

                    self.ckpt_manager.save(
                        self.agent,
                        episode,
                        self.agent.total_steps,
                        best_reward,
                        avg_rw,
                        is_best=is_best,
                    )

            recent_rewards.append(episode_reward)
            avg_50 = float(np.mean(recent_rewards[-50:]))

            # Registros ao final de cada episódio
            self.writer.add_scalar("Episode/Reward", episode_reward, episode)
            self.writer.add_scalar("Episode/Average_Reward_50", avg_50, episode)
            self.writer.add_scalar("Episode/Steps", episode_steps, episode)
            self.writer.add_scalar("Episode/Final_Speed", info.get("speed", 0.0), episode)

        pbar.close()
        self.writer.close()
        print("[Trainer] Treinamento concluído com sucesso!")