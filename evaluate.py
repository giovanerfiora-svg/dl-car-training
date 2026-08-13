import argparse
import yaml
import torch
from training.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Avaliar Desempenho do Agente DRL")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Caminho do modelo para avaliar")
    parser.add_argument("--episodes", type=int, default=10, help="Quantidade de episódios para teste")
    args = parser.parse_args()

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config)

    try:
        checkpoint = torch.load(args.model, map_location=evaluator.agent.device)
        evaluator.agent.q_policy.load_state_dict(checkpoint["q_policy_state_dict"])
        print(f"[Evaluate] Pesos carregados do modelo: {args.model}")
    except Exception as e:
        print(f"[Evaluate] Falha ao carregar modelo '{args.model}': {e}")
        return

    print(f"\n--- Iniciando Avaliação em {args.episodes} Episódios ---")
    metrics = evaluator.evaluate(num_episodes=args.episodes)

    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")


if __name__ == "__main__":
    main()