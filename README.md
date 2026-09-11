# 🏎️ DRL Autonomous Car Lab (DQN)

Projeto de **Deep Reinforcement Learning** em Python no qual um agente aprende a dirigir um carro em um circuito 2D. O projeto usa **DQN (Deep Q-Network)** e foi organizado para facilitar experimentação, observação das métricas e evolução do agente.

## 🧠 O que o agente aprende

O agente recebe um vetor de observações formado por:

- 7 sensores LiDAR de proximidade;
- velocidade normalizada;
- posição relativa do próximo checkpoint.

Ele escolhe entre 5 ações discretas: **manter, acelerar, frear, esquerda e direita**.

A recompensa combina progresso no circuito, velocidade, penalidade por ficar parado e penalidade por colisão.

## 🔬 Algoritmo

A implementação utiliza:

- **PyTorch** para as redes neurais;
- **Replay Buffer** para reutilizar experiências;
- **Target Network** para estabilizar os alvos de treinamento;
- **Double DQN** para reduzir a superestimação dos valores Q;
- **Huber Loss (SmoothL1Loss)** para tornar o treinamento menos sensível a erros grandes;
- **Gradient Clipping** para evitar atualizações exageradas;
- **Epsilon-Greedy** para equilibrar exploração e aproveitamento.

## 📊 Avaliação quantitativa

O projeto possui um avaliador separado do treinamento. Ele executa episódios usando a política gulosa (`epsilon=0`) e calcula:

- recompensa média e desvio padrão;
- melhor recompensa;
- passos médios por episódio;
- checkpoints médios e máximos;
- taxa de colisão;
- velocidade média;
- quantidade de passos de treinamento presentes no checkpoint.

Execute:

```bash
python evaluation.py --model models/best.pt --episodes 20
```

O resultado também é salvo em `evaluation_results.json`, facilitando comparações entre versões do agente.

## 📁 Estrutura

```text
.
├── agent/
│   ├── dqn_agent.py
│   ├── network.py
│   └── replay_buffer.py
├── configs/
│   └── default.yaml
├── environment/
│   ├── car.py
│   ├── track.py
│   └── vehicle_env.py
├── models/
├── training/
│   ├── checkpoint.py
│   └── trainer.py
├── evaluation.py
├── evaluate.py
├── play.py
└── train.py
```

## 🚀 Instalação

```bash
pip install -r requirements.txt
```

## ▶️ Treinamento

```bash
python train.py
```

O treinador retoma automaticamente o `models/latest.pt` quando existe um checkpoint compatível.

## 🎮 Visualização

```bash
python play.py
```

## 📈 TensorBoard

```bash
tensorboard --logdir runs
```

## 🧪 Próximas evoluções

- testes automatizados do ambiente e do agente;
- gráfico automático dos resultados de avaliação;
- comparação DQN vs. Double DQN usando o mesmo protocolo;
- priorização de experiências no replay buffer;
- exploração de **Dueling DQN**;
- melhoria progressiva da função de recompensa;
- avaliação de generalização em circuitos diferentes.

## 📄 Licença

MIT
