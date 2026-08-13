import yaml
from training.trainer import Trainer


def main():
    # Carregar configurações centralizadas
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Instanciar e rodar o treinador
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()