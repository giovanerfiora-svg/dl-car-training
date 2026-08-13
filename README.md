# 🏎️ DRL Autonomous Car Lab (DQN)

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame--CE-green.svg)
![License](https://img.shields.io/badge/license-MIT-informational.svg)

Projeto de Aprendizado por Reforço Profundo (Deep Reinforcement Learning) em Python onde um agente inteligente aprende a pilotar um veículo autônomo em um simulador 2D usando um algoritmo **Deep Q-Network (DQN)**.

## 🚀 Tecnologias Utilizadas
* **Python 3.14+**
* **PyTorch** (Redes Neurais / Q-Network)
* **Gymnasium** (Ambiente de Aprendizado por Reforço)
* **Pygame-CE** (Renderização visual em tempo real)
* **TensorBoard** (Monitoramento de métricas e convergência)

## 🧠 Arquitetura do Agente
* **Sensores:** 7 Raios LiDAR de proximidade + Velocidade + Ângulo do Volante (Vetor de dimensão 9).
* **Ações:** 5 Ações discretas (Acelerar, Frear, Esquerda, Direita, Manter).
* **Algoritmo:** DQN com *Replay Buffer* descorrelacionado e *Target Network* sincronizada periodicamente.

## 🛠️ Como Executar

1. Instale as dependências:
   ```bash
   pip install torch pygame-ce tensorboard matplotlib pytest pyyaml tqdm
   ```

2. Inicie o treinamento:
   ```bash
   python train.py
   ```

3. Assista ao agente em tempo real:
   ```bash
   python play.py
   ```
