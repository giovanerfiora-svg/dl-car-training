from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import yaml

from agent.dqn_agent import DQNAgent
from environment.vehicle_env import VehicleEnv


def load_checkpoint(agent: DQNAgent, path: str) -> dict[str, Any]:
    """Carrega somente os pesos necessários para avaliação, sem otimização."""
    checkpoint = torch.load(path, map_location=agent.device)
    agent.q_policy.load_state_dict(checkpoint["q_policy_state_dict"])
    agent.q_policy.eval()
    return checkpoint


def evaluate(config: dict[str, Any], model_path: str, episodes: int) -> dict[str, float]:
    env = VehicleEnv(config)
    agent = DQNAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        config,
    )
    checkpoint = load_checkpoint(agent, model_path)

    rewards: list[float] = []
    steps: list[int] = []
    checkpoints: list[int] = []
    crashes: list[int] = []
    speeds: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=config["project"]["seed"] + episode)
        done = False
        total_reward = 0.0
        episode_steps = 0
        max_checkpoint = 0
        crashed = 0
        speed_sum = 0.0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_steps += 1
            max_checkpoint = max(max_checkpoint, int(info.get("checkpoint", 0)))
            crashed = int(info.get("is_crashed", False))
            speed_sum += float(info.get("speed", 0.0))

        rewards.append(total_reward)
        steps.append(episode_steps)
        checkpoints.append(max_checkpoint)
        crashes.append(crashed)
        speeds.append(speed_sum / max(episode_steps, 1))

        print(
            f"Episode {episode + 1:03d} | reward={total_reward:8.2f} | "
            f"steps={episode_steps:4d} | checkpoints={max_checkpoint:3d} | "
            f"crashed={crashed}"
        )

    results = {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps)),
        "mean_checkpoints": float(np.mean(checkpoints)),
        "max_checkpoints": float(np.max(checkpoints)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(speeds)),
        "checkpoint_training_steps": float(checkpoint.get("total_steps", 0)),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia um agente DQN treinado.")
    parser.add_argument("--model", default="models/best.pt", help="Checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=20, help="Número de episódios")
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Arquivo JSON de saída",
    )
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Modelo não encontrado: {args.model}")
    if args.episodes <= 0:
        raise ValueError("--episodes precisa ser maior que zero.")

    results = evaluate(config, args.model, args.episodes)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\n=== Resultado da avaliação ===")
    for name, value in results.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
