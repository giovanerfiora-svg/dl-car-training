import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from environment.car import Car
from environment.track import Track


class VehicleEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        env_cfg = config["environment"]

        self.track = Track(name=env_cfg["track_name"])
        self.car = Car(
            x=250,
            y=100,
            angle=0.0,
            max_speed=env_cfg["car_max_speed"],
            acceleration=env_cfg["car_acceleration"],
            steering_speed=env_cfg["car_steering_speed"],
            num_lidar_rays=env_cfg["num_lidar_rays"],
            lidar_max_dist=env_cfg["lidar_max_dist"],
        )

        # Ações Discretas: [0: Manter, 1: Acelerar, 2: Frear, 3: Esquerda, 4: Direita]
        self.action_space = spaces.Discrete(5)

        # Espaço de Observação: [Velocidade Normalizada, N Raios LiDAR Normalizados, Progresso Próximo Checkpoint]
        obs_dim = 1 + env_cfg["num_lidar_rays"] + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = env_cfg["max_steps"]
        self.current_checkpoint_idx = 0
        self.reward_weights = config["reward_weights"]

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.car.reset()
        self.current_step = 0
        self.current_checkpoint_idx = 0

        obs = self._get_observation()
        info = {"checkpoint": self.current_checkpoint_idx}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # 1. Atualizar Física
        self.car.update(action)

        # 2. Atualizar Sensores
        lidar_dists = self.car.cast_lidar_rays(self.track.walls)

        # 3. Checar Colisão
        is_crashed = self.track.check_collision(self.car.x, self.car.y)
        self.car.is_crashed = is_crashed

        # 4. Calcular Recompensa
        reward, reached_checkpoint = self._calculate_reward(is_crashed)

        # 5. Condição de Término
        terminated = is_crashed
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "is_crashed": is_crashed,
            "step": self.current_step,
            "speed": self.car.speed,
            "checkpoint": self.current_checkpoint_idx,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        norm_speed = self.car.speed / self.car.max_speed
        norm_lidar = [
            (d / self.car.lidar_max_dist) * 2.0 - 1.0 for d in self.car.lidar_distances
        ]
        next_ckpt_idx = (self.current_checkpoint_idx + 1) % len(self.track.checkpoints)
        norm_ckpt = (next_ckpt_idx / len(self.track.checkpoints)) * 2.0 - 1.0

        obs = [norm_speed] + norm_lidar + [norm_ckpt]
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, is_crashed: bool) -> Tuple[float, bool]:
        rw = self.reward_weights
        if is_crashed:
            return rw["collision_penalty"], False

        # Recompensa por velocidade (incentiva andar para a frente)
        reward = (self.car.speed / self.car.max_speed) * rw["speed_bonus"]

        # Penalidade por ficar parado
        if abs(self.car.speed) < 0.1:
            reward += rw["idle_penalty"]

        # Recompensa ao passar por checkpoint
        reached_checkpoint = False
        next_ckpt_idx = (self.current_checkpoint_idx + 1) % len(self.track.checkpoints)
        ckpt_line = self.track.checkpoints[next_ckpt_idx]

        # Checar se o carro atingiu a linha do próximo checkpoint
        if self._car_crossed_line(ckpt_line):
            self.current_checkpoint_idx = next_ckpt_idx
            reward += rw["progress"]
            reached_checkpoint = True

        return reward, reached_checkpoint

    def _car_crossed_line(self, line: Tuple[float, float, float, float]) -> bool:
        dist = np.hypot(self.car.x - (line[0] + line[2]) / 2, self.car.y - (line[1] + line[3]) / 2)
        return dist < 40.0