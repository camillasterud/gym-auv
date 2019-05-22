from copy import deepcopy
import numpy as np
import numpy.linalg as linalg

from scipy import interpolate
from scipy.optimize import fminbound

import gym_auv.utils.geomutils as geom

class ParamCurve():
    def __init__(self, waypoints):

        for _ in range(3):
            arclengths = arc_len(waypoints)
            path_coords = interpolate.pchip(
                x=arclengths, y=waypoints, axis=1)
            waypoints = path_coords(
                np.linspace(arclengths[0], arclengths[-1], 1000))

        self.path_coords = path_coords
        self.s_max = arclengths[-1]
        self.length = self.s_max

    def __call__(self, arclength):
        return self.path_coords(arclength)

    def get_direction(self, arclength):
        arclength = np.clip(arclength, 0.05, self.s_max - 0.05)
        delta_x, delta_y = (self.path_coords(arclength + 0.05)
                            - self.path_coords(arclength - 0.05))
        return geom.princip(np.arctan2(delta_y, delta_x))

    def get_endpoint(self):
        return self(self.s_max)

    def get_closest_arclength(self, position):
        return fminbound(lambda s: linalg.norm(self(s) - position),
                         x1=0, x2=self.length, xtol=1e-6,
                         maxfun=10000)

    def __reversed__(self):
        curve = deepcopy(self)
        path_coords = curve.path_coords
        curve.path_coords = lambda s: path_coords(curve.length-s)
        return curve

    def plot(self, ax, s, *opts):
        s = np.array(s)
        z = self(s)
        ax.plot(-z[1, :], z[0, :], *opts)

class RandomCurveThroughOrigin(ParamCurve):
    def __init__(self, rng, nwaypoints, length=400):
        angle_init = 2*np.pi*(rng.rand() - 0.5)
        start = np.array([length*np.cos(angle_init),
                          length*np.sin(angle_init)])
        end = -np.array(start)
        waypoints = np.vstack([start, end])
        for waypoint in range(nwaypoints // 2):
            newpoint1 = ((nwaypoints // 2 - waypoint)
                         * start / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            newpoint2 = ((nwaypoints // 2 - waypoint)
                         * end / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            waypoints = np.vstack([waypoints[:waypoint+1, :],
                                   newpoint1,
                                   np.array([0, 0]),
                                   newpoint2,
                                   waypoints[-1*waypoint-1:, :]])
        super().__init__(np.transpose(waypoints))


def arc_len(coords):
    diff = np.diff(coords, axis=1)
    delta_arc = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.concatenate([[0], np.cumsum(delta_arc)])
