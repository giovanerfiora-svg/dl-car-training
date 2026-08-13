import yaml
import pytest
import numpy as np
from environment.vehicle_env import VehicleEnv


@pytest.fixture
def config():
    with open("configs/default.yaml", "r") as f:
        return yaml.safe_load(f)


def test_env_initialization(config):
    env = VehicleEnv(config)
    obs, info = env.reset(seed=42)

    assert obs.shape == (9,)
    assert isinstance(obs, np.ndarray)
    assert info["checkpoint"] == 0


def test_env_step_collision(config):
    env = VehicleEnv(config)
    env.reset(seed=42)

    done = False
    steps = 0
    while not done and steps < 200:
        obs, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        steps += 1

    assert steps < 200
    assert info["is_crashed"] is True
    assert reward <= config["reward_weights"]["collision_penalty"]