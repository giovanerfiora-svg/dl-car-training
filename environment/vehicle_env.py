import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple

from environment.car import Car
from environment.track import Track


class VehicleEnv(gym.Env):
    """Ambiente 2D de direção usado pelo agente DQN."""

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

        self.action_space = spaces.Discrete(5)

        obs_dim = 1 + env_cfg["num_lidar_rays"] + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = int(env_cfg["max_steps"])
        self.current_checkpoint_idx = 0
        self.reward_weights = config["reward_weights"]

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.car.reset()
        self.current_step = 0
        self.current_checkpoint_idx = 0

        obs = self._get_observation()
        return obs, {"checkpoint": self.current_checkpoint_idx}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self.car.update(int(action))

        self.car.cast_lidar_rays(self.track.walls)
        is_crashed = self.track.check_collision(self.car.x, self.car.y)
        self.car.is_crashed = is_crashed

        reward, reached_checkpoint = self._calculate_reward(is_crashed)
        terminated = is_crashed
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = {
            "is_crashed": is_crashed,
            "reached_checkpoint": reached_checkpoint,
            "step": self.current_step,
            "speed": self.car.speed,
            "checkpoint": self.current_checkpoint_idx,
        }
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        norm_speed = np.clip(self.car.speed / self.car.max_speed, 0.0, 1.0)
        norm_lidar = [
            np.clip((d / self.car.lidar_max_dist) * 2.0 - 1.0, -1.0, 1.0)
            for d in self.car.lidar_distances
        ]
        checkpoint_count = len(self.track.checkpoints)
        norm_ckpt = (
            (self.current_checkpoint_idx / max(checkpoint_count - 1, 1)) * 2.0 - 1.0
        )

        return np.asarray(
            [norm_speed] + norm_lidar + [norm_ckpt], dtype=np.float32
        )

    def _calculate_reward(self, is_crashed: bool) -> Tuple[float, bool]:
        rw = self.reward_weights
        if is_crashed:
            return float(rw["collision_penalty"]), False

        reward = (self.car.speed / self.car.max_speed) * rw["speed_bonus"]

        if abs(self.car.speed) < 0.1:
            reward += rw["idle_penalty"]

        reached_checkpoint = False
        next_ckpt_idx = (self.current_checkpoint_idx + 1) % len(self.track.checkpoints)
        ckpt_line = self.track.checkpoints[next_ckpt_idx]

        if self._car_crossed_line(ckpt_line):
            self.current_checkpoint_idx = next_ckpt_idx
            reward += rw["progress"]
            reached_checkpoint = True

        return float(reward), reached_checkpoint

    def _car_crossed_line(self, line: Tuple[float, float, float, float]) -> bool:
        dist = np.hypot(
            self.car.x - (line[0] + line[2]) / 2,
            self.car.y - (line[1] + line[3]) / 2,
        )
        return bool(dist < 40.0)
