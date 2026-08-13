import argparse
import yaml
import torch
from environment.vehicle_env import VehicleEnv
from agent.dqn_agent import DQNAgent
from visualization.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description="Visualizar Agente em Tempo Real")
    parser.add_argument("--model", type=str, default="models/latest.pt", help="Caminho do modelo .pt")
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = VehicleEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config)

    # Carregar modelo salvo
    try:
        checkpoint = torch.load(args.model, map_location=agent.device)
        agent.q_policy.load_state_dict(checkpoint["q_policy_state_dict"])
        agent.q_policy.eval()
        print(f"[Play] Modelo '{args.model}' carregado com sucesso!")
    except Exception as e:
        print(f"[Play] Erro ao carregar modelo '{args.model}': {e}")
        print("[Play] Executando com pesos aleatórios para demonstração.")

    renderer = Renderer()
    running = True

    while running:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done and running:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            info_str = f"Speed: {env.car.speed:.1f} | Reward: {episode_reward:.1f} | Checkpoint: {info['checkpoint']}"
            running = renderer.render(
                car_x=env.car.x,
                car_y=env.car.y,
                car_angle=env.car.angle,
                is_crashed=info.get("is_crashed", False),
                lidar_distances=env.car.lidar_distances,
                walls=env.track.walls,
                checkpoints=env.track.checkpoints,
                info_text=info_str,
            )

    renderer.close()


if __name__ == "__main__":
    main()