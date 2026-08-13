import math
from typing import List, Tuple
import numpy as np

class Car:
    def __init__(
            self,
            x: float,
            y: float,
            angle: float = 0.0,
            max_speed: float = 8.0,
            acceleration: float = 0.2,
            steering_speed: float = 4.0,
            num_lidar_rays: int = 7,
            lidar_max_dist: float = 200,
    ):
        self.initial_x = x
        self.initial_y = y
        self.initial_angle = angle

        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0

        self.max_speed = max_speed
        self.acceleration = acceleration
        self.steering_speed = steering_speed

        self.num_lidar_rays = num_lidar_rays
        self.lidar_max_dist = lidar_max_dist
        self.lidar_distances = [lidar_max_dist] * num_lidar_rays

        self.width = 20
        self.height = 10
        self.is_crashed = False

    def reset(self) -> None:
        self.x = self.initial_x
        self.y = self.initial_y
        self.angle = self.initial_angle
        self.speed = 0.0
        self.is_crashed = False
        self.lidar_distances = [self.lidar_max_dist] * self.num_lidar_rays

    def update(self, action: int) -> None:

        if action == 1: # Acelerar
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif action == 2: # Frear
            self.speed = max(self.speed - self.acceleration, -self.max_speed / 2.0)
        elif action == 3: # Esquerda
            self.angle -= self.steering_speed
        elif action == 4: # Direita
            self.angle += self.steering_speed

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

    def cast_lidar_rays(self, track_walls: List[Tuple[float, float, float, float]]) -> List[float]:
        field_of_view = 160.0  # Leque de 160 graus na frente do carro
        start_angle = self.angle - (field_of_view / 2.0)
        angle_step = field_of_view / (self.num_lidar_rays - 1) if self.num_lidar_rays > 1 else 0

        self.lidar_distances = []

        for i in range(self.num_lidar_rays):
            ray_angle = math.radians(start_angle + (i * angle_step))
            dist = self._cast_single_ray(ray_angle, track_walls)
            self.lidar_distances.append(dist)

        return self.lidar_distances

    def _cast_single_ray(
            self, ray_angle: float, walls: List[Tuple[float, float, float, float]]
    ) -> float:
        min_dist = self.lidar_max_dist
        x1, y1 = self.x, self.y
        x2 = self.x + self.lidar_max_dist * math.cos(ray_angle)
        y2 = self.y + self.lidar_max_dist * math.sin(ray_angle)

        for wall in walls:
            x3, y3, x4, y4 = wall
            intersection = self._line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
            if intersection:
                dist = math.hypot(intersection[0] - x1, intersection[1] - y1)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    @staticmethod
    def _line_intersection(
        line1: Tuple[float, float, float, float], line2: Tuple[float, float, float, float]
    ) -> Tuple[float, float] | None:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None