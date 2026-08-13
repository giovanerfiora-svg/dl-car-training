import os
import yaml
import pytest
import torch
from agent.dqn_agent import DQNAgent
from training.checkpoint import CheckpointManager


@pytest.fixture
def config():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_checkpoint_save_and_load(config, tmp_path):
    save_dir = str(tmp_path / "models")
    ckpt_manager = CheckpointManager(save_dir=save_dir)
    agent = DQNAgent(state_dim=9, action_dim=5, config=config)

    # Modificar o epsilon e os passos do agente
    agent.epsilon = 0.42
    agent.total_steps = 1337

    # Salvar o estado
    ckpt_manager.save(
        agent=agent,
        episode=10,
        total_steps=1337,
        best_reward=150.0,
        recent_avg_reward=120.0,
        is_best=True,
    )

    # Resetar o agente para conferir se a recarga restaura os valores
    new_agent = DQNAgent(state_dim=9, action_dim=5, config=config)
    episode, steps, best_rw, loaded = ckpt_manager.load_latest(new_agent)

    assert loaded is True
    assert episode == 10
    assert steps == 1337
    assert best_rw == 150.0
    assert pytest.approx(new_agent.epsilon) == 0.42
    assert os.path.exists(ckpt_manager.best_path)