from typing import List, Tuple
import numpy as np


class Track:
    
    def __init__(self, name: str = "circuit_basic"):
        self.name = name
        self.walls: List[Tuple[float, float, float, float]] = []
        self.checkpoints: List[Tuple[float, float, float, float]] = []
        self._build_track()

    def _build_track(self) -> None:
        # Paredes Externas (X1, Y1, X2, Y2)
        outer = [
            (50, 50, 750, 50),
            (750, 50, 750, 550),
            (750, 550, 50, 550),
            (50, 550, 50, 50),
        ]
        # Paredes Internas
        inner = [
            (150, 150, 650, 150),
            (650, 150, 650, 450),
            (650, 450, 150, 450),
            (150, 450, 150, 150),
        ]
        self.walls = outer + inner

        # Checkpoints de Progresso ao longo do circuito
        self.checkpoints = [
            (400, 50, 400, 150),   # Linha de Chegada / Checkpoint 0
            (750, 300, 650, 300),  # Checkpoint 1
            (400, 550, 400, 450),  # Checkpoint 2
            (50, 300, 150, 300),   # Checkpoint 3
        ]

    def check_collision(self, car_x: float, car_y: float, radius: float = 8.0) -> bool:
        for x1, y1, x2, y2 in self.walls:
            # Distância de um ponto a um segmento de reta
            px = x2 - x1
            py = y2 - y1
            norm = px * px + py * py
            if norm == 0:
                continue
            u = max(0, min(1, ((car_x - x1) * px + (car_y - y1) * py) / float(norm)))
            ix = x1 + u * px
            iy = y1 + u * py
            dist = float(np.hypot(car_x - ix, car_y - iy))
            if dist < radius:
                return True
        return False