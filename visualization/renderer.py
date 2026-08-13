import math
from typing import List, Tuple
import pygame


class Renderer:

    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        pygame.display.set_caption("DL Autonomous Car Lab")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Cores (RGB)
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_WALL = (200, 200, 200)
        self.COLOR_CAR = (0, 255, 128)
        self.COLOR_CAR_CRASHED = (255, 50, 50)
        self.COLOR_LIDAR = (255, 255, 0)
        self.COLOR_CHECKPOINT = (0, 150, 255)

    def render(
        self,
        car_x: float,
        car_y: float,
        car_angle: float,
        is_crashed: bool,
        lidar_distances: List[float],
        walls: List[Tuple[float, float, float, float]],
        checkpoints: List[Tuple[float, float, float, float]],
        info_text: str = "",
    ) -> bool:
        """Desenha um quadro da simulação. Retorna False se o usuário fechar a janela."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(self.COLOR_BG)

        # 1. Desenhar Paredes da Pista
        for x1, y1, x2, y2 in walls:
            pygame.draw.line(self.screen, self.COLOR_WALL, (int(x1), int(y1)), (int(x2), int(y2)), 3)

        # 2. Desenhar Checkpoints
        for x1, y1, x2, y2 in checkpoints:
            pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (int(x1), int(y1)), (int(x2), int(y2)), 1)

        # 3. Desenhar Raios do LiDAR
        num_rays = len(lidar_distances)
        if num_rays > 0:
            field_of_view = 160.0
            start_angle = car_angle - (field_of_view / 2.0)
            angle_step = field_of_view / (num_rays - 1) if num_rays > 1 else 0

            for i, dist in enumerate(lidar_distances):
                ray_angle = math.radians(start_angle + (i * angle_step))
                end_x = car_x + dist * math.cos(ray_angle)
                end_y = car_y + dist * math.sin(ray_angle)
                pygame.draw.line(
                    self.screen, self.COLOR_LIDAR, (int(car_x), int(car_y)), (int(end_x), int(end_y)), 1
                )

        # 4. Desenhar Veículo
        color = self.COLOR_CAR_CRASHED if is_crashed else self.COLOR_CAR
        pygame.draw.circle(self.screen, color, (int(car_x), int(car_y)), 10)

        # Indicador de direção do carro
        rad = math.radians(car_angle)
        dir_x = car_x + 15 * math.cos(rad)
        dir_y = car_y + 15 * math.sin(rad)
        pygame.draw.line(self.screen, (255, 255, 255), (int(car_x), int(car_y)), (int(dir_x), int(dir_y)), 2)

        # 5. Overlay de Texto com Métricas
        if info_text:
            font = pygame.font.SysFont("Consolas", 16)
            surface = font.render(info_text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)  # Limitar a 30 FPS
        return True

    def close(self) -> None:
        pygame.quit()