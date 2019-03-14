import numpy as np

class StaticObstacle():
    def __init__(self, position, radius, color=(0.6, 0, 0)):
        self.color = color
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if radius < 0:
            raise ValueError
        self.radius = radius
        self.position = position.flatten()

    def step(self):
        pass

    def draw(self, viewer, color=None):
        viewer.draw_circle(self.position, self.radius,
                           color=self.color if color is None else color)
