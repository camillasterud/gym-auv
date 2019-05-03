import sys

import numpy as np
import numpy.linalg as linalg

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom

class AUV2D():
    def __init__(self, t_step, init_pos, width=2):

        self._state = np.hstack([init_pos, [0, 0, 0]])
        self.path_taken = np.array([init_pos[0:2]])
        self.color = (0.6, 0.6, 0.6)
        self.radius = width
        self.t_step = t_step
        self.input = [0, 0]
        self.vertices = [
            (-width, -width),
            (-width, width),
            (2 * width, width),
            (3 * width, 0),
            (2 * width, -width),
        ]

    def step(self, action):
        self.input = np.array([_surge(action[0]), _steer(action[1])])
        self._sim()

        if linalg.norm(self.position - self.path_taken[-1]) > 3:
            self.path_taken = np.vstack([self.path_taken, self.position])

    def draw(self, viewer):
        position = self.position
        angle = self.heading
        viewer.draw_polyline(self.path_taken, linewidth=3, color=(0.8, 0, 0))  # previous positions
        viewer.draw_shape(self.vertices, position, angle, self.color)  # ship
        viewer.draw_arrow(position, angle + np.pi + self.input[1]/4, length=2)

    def _sim(self):
        _, _, psi, u, v, r = self._state
        nu = np.array([u, v, r])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(const.B(u).dot(self.input) - const.D(u, v, r).dot(nu))
        state_dot = np.concatenate([eta_dot, nu_dot])
        self._state += state_dot*self.t_step

    @property
    def position(self):
        return self._state[0:2]
    
    @property
    def heading(self):
        return self._state[2]
    
    @property
    def velocity(self):
        return self._state[3:5]

    @property
    def yawrate(self):
        return self._state[5]
    

    @property
    def max_speed(self):
        return const.MAX_SPEED


def _surge(surge):
    surge = np.clip(surge, 0, 1)
    return surge*(const.THRUST_MAX_AUV - const.THRUST_MIN_AUV) + const.THRUST_MIN_AUV

def _steer(steer):
    steer = np.clip(steer, -1, 1)
    return steer*const.RUDDER_MAX_AUV
