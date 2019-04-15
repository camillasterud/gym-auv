"""
This module implements an AUV that is simulated in the horizontal plane.
"""
import numpy as np
import numpy.linalg as linalg

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom

class AUV2D():
    """
    Creates an environment with a vessel, goal and obstacles.

    Attributes
    ----------
    path_taken : np.array
        Array of size (?, 2) discribing the path the AUV has taken.
    radius : float
        The maximum distance from the center of the AUV to its edge
        in meters.
    t_step : float
        The simulation timestep.
    input : np.array
        The current input. [propeller_input, rudder_position].
    """
    def __init__(self, t_step, init_pos, width=1):
        """
        The __init__ method declares all class atributes.

        Parameters
        ----------
        t_step : float
            The simulation timestep to be used to simulate this AUV.
        init_pos : np.array
            The initial position of the vessel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The maximum distance from the center of the AUV to its edge
            in meters. Defaults to 1.
        """
        self._state = np.hstack([init_pos, [0, 0, 0]])
        self.path_taken = np.array([init_pos[0:2]])
        self.radius = width
        self.t_step = t_step
        self.input = [0, 0]

    def step(self, action):
        """
        Steps the vessel self.t_step seconds forward.

        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position], where
            0 <= propeller_input <= 1 and -1 <= rudder_position <= 1.
        """
        self.input = np.array([_surge(action[0]), _steer(action[1])])
        self._sim()

        if linalg.norm(self.position - self.path_taken[-1]) > 1:
            self.path_taken = np.vstack([self.path_taken,
                                         self.position])

    def _sim(self):
        psi = self._state[2]
        nu = self._state[3:]

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(const.B(nu).dot(self.input)
                                 - const.D(nu).dot(nu)
                                 - const.C(nu).dot(nu)
                                 - const.L(nu).dot(nu))
        state_dot = np.concatenate([eta_dot, nu_dot])
        self._state += state_dot*self.t_step

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in cartesian
        coordinates.
        """
        return self._state[0:2]

    @property
    def heading(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self._state[2]

    @property
    def velocity(self):
        """
        Returns the surge and sway velocity of the AUV.
        """
        return self._state[3:5]

    @property
    def max_speed(self):
        """
        Returns the max speed of the AUV.
        """
        return const.MAX_SPEED


def _surge(surge):
    surge = np.clip(surge, 0, 1)
    return (surge*(const.THRUST_MAX_AUV - const.THRUST_MIN_AUV)
            + const.THRUST_MIN_AUV)

def _steer(steer):
    steer = np.clip(steer, -1, 1)
    return steer*const.RUDDER_MAX_AUV
