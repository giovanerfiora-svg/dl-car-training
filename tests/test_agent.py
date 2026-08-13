import yaml
import pytest
import numpy as np
from agent.dqn_agent import DQNAgent


@pytest.fixture
def config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_agent_action_selection(config):
    agent = DQNAgent(state_dim=9, action_dim=5, config=config)
    fake_state = np.zeros(9, dtype=np.float32)

    action = agent.select_action(fake_state, evaluate=True)
    assert 0 <= action < 5


def test_agent_update_step(config):
    agent = DQNAgent(state_dim=9, action_dim=5, config=config)
    fake_state = np.zeros(9, dtype=np.float32)

    # Preencher o buffer até o threshold mínimo de treino
    for _ in range(config["agent"]["train_start_size"] + 10):
        agent.replay_buffer.push(fake_state, 1, 1.0, fake_state, False)

    loss = agent.update()
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0.0