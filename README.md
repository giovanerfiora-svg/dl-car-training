# 🏎️ DRL Autonomous Car Lab (Dueling Double DQN)

Projeto de **Deep Reinforcement Learning** em Python no qual um agente aprende a dirigir um carro em um circuito 2D.

## 🧠 Agente

O agente observa:

- 7 sensores LiDAR;
- velocidade normalizada;
- posição relativa do checkpoint.

Ele escolhe entre 5 ações discretas: manter, acelerar, frear, esquerda e direita.

A implementação combina **Double DQN + Dueling DQN + Huber Loss**.

### Dueling DQN

A rede compartilhada gera duas estimativas:

- `V(s)`: quão bom é o estado atual;
- `A(s,a)`: quão boa é cada ação naquele estado.

Depois combinamos as duas:

```text
Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
```

Isso ajuda o agente a aprender situações em que várias ações têm valor parecido, separando o valor do estado da vantagem de escolher uma ação específica.

## 📊 Métricas de treinamento

O TensorBoard registra, por passo:

- `Train/Loss`
- `Train/Epsilon`
- `Train/ReplayBufferSize`
- `Train/LearningRate`
- `Train/GradientNorm`

E, por episódio:

- recompensa;
- recompensa média dos últimos 50 episódios;
- quantidade de passos;
- velocidade média;
- checkpoints alcançados;
- colisões;
- loss médio;
- taxa de conclusão do episódio.

## 📈 TensorBoard

```bash
tensorboard --logdir runs
```

A ideia é acompanhar não apenas se a recompensa sobe, mas também **como** o agente está aprendendo: se o loss estabiliza, se a exploração diminui, se o buffer cresce e se colisões diminuem.

## 🧪 Avaliação

A avaliação ocorre separadamente do treinamento usando política gulosa:

```bash
python evaluation.py --model models/best.pt --episodes 20
```

O avaliador calcula recompensa média, desvio padrão, melhor recompensa, passos médios, checkpoints, taxa de colisão e velocidade média.

## 🚀 Executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Treine:

```bash
python train.py
```

Visualize:

```bash
python play.py
```

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
├── training/
│   ├── checkpoint.py
│   └── trainer.py
├── evaluation.py
├── evaluate.py
├── play.py
└── train.py
```

## 🔬 Próximas evoluções

- testes automatizados;
- comparação experimental DQN vs. Double DQN vs. Dueling Double DQN;
- Prioritized Experience Replay;
- avaliação em circuitos diferentes;
- gráficos automáticos de desempenho;
- estudo do impacto de diferentes funções de recompensa.

## 📄 Licença

MIT
