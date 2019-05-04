"""
This module implements the AUV gym environment through the AUVenv class.
"""

import gym
from gym.utils import seeding
import numpy as np
import numpy.linalg as linalg

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin
from gym_auv.objects.obstacles import StaticObstacle

class PathFollowingEnv(gym.Env):
    """
    Creates an environment with a vessel, path and obstacles.
    Attributes
    ----------
    config : dict
        The configuration disctionary specifying rewards,
        number of obstacles, LOS distance, obstacle detection range,
        simulation timestep and desired cruising speed.
    nstates : int
        The number of state variables passed to the agent.
    nsectors : int
        The number of obstacle detection sectors around the vessel.
    vessel : gym_auv.objects.auv.AUV2D
        The AUV that is controlled by the agent.
    path : gym_auv.objects.path.RandomCurveThroughOrigin
        The path to be followed.
    np_random : np.random.RandomState
        Random number generator.
    obstacles : list
        List of obstacles.
    reward : float
        The accumulated reward
    path_prog : float
        Progression along the path in terms of arc length covered.
    last_action : np.array
        The last action that was preformed.
    action_space : gym.spaces.Box
        The action space. Consists of two floats that must take on
        values between -1 and 1.
    observation_space : gym.spaces.Box
        The observation space. Consists of
        self.nstates + self.nsectors floats that must be between
        0 and 1.
    """

    metadata = {}

    def __init__(self, env_config):
        """
        The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
        env_config : dict
            Configuration parameters for the environment.
            Must have the following members:
            reward_ds
                The reward for progressing ds along the path in
                one timestep. reward += reward_ds*ds.
            reward_speed_error
                reward += reward_speed_error*speed_error where the
                speed error is abs(speed-cruise_speed)/max_speed.
            reward_cross_track_error
                reward += reward_cross_track_error*cross_track_error
            los_dist
                The line of sight distance.
            t_step
                The timestep
            cruise_speed
                The desired cruising speed.
        """
        self.config = env_config
        self.nstates = 7
        nactions = 2
        nobservations = self.nstates
        self.vessel = None
        self.path = None

        self.np_random = None
        self.obstacles = None

        self.reward = 0
        self.path_prog = 0
        self.last_action = None

        self.reset()

        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*nobservations),
            high=np.array([1]*nobservations),
            dtype=np.float32)

    def step(self, action):
        """
        Simulates the environment for one timestep when action
        is performed

        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position].

        Returns
        -------
        obs : np.array
            Observation of the environment after action is performed.
        step_reward : double
            The reward for performing action at his timestep.
        done : bool
            If True the episode is ended.
        info : dict
            Empty, is included because it is required of the
            OpenAI Gym frameowrk.
        """
        self.last_action = action
        self.vessel.step(action)

        prog = self.path.get_closest_arclength(self.vessel.position)
        delta_path_prog = prog - self.path_prog
        self.path_prog = prog

        obs = self.observe()
        done, step_reward = self.step_reward(obs, delta_path_prog)
        info = {}

        return obs, step_reward, done, info

    def step_reward(self, obs, delta_path_prog):
        """
        Calculates the step_reward and decides whether the episode
        should be ended.

        Parameters
        ----------
        obs : np.array
            The observation of the environment.
        delta_path_prog : double
            How much the vessel has cavered of the path arclength
            the last timestep.
        Returns
        -------
        done : bool
            If True the episode is ended.
        step_reward : double
            The reward for performing action at his timestep.
        """
        done = False
        step_reward = 0

        step_reward += delta_path_prog*self.config["reward_ds"]
        speed_error = ((linalg.norm(self.vessel.velocity)
                        - self.config["cruise_speed"])
                       /self.vessel.max_speed)
        cross_track_error = obs[4]

        step_reward += (abs(cross_track_error)
                        *self.config["reward_cross_track_error"])
        step_reward += (abs(speed_error)
                        *self.config["reward_speed_error"])

        dist_to_endpoint = linalg.norm(self.vessel.position
                                       - self.path.get_endpoint())

        self.reward += step_reward

        if (self.reward < self.config["min_reward"]
                or abs(self.path_prog - self.path.length) < 1
                or dist_to_endpoint < 5):
            done = True

        return done, step_reward

    def generate(self):
        """
        Sets up the environment. Places the goal and obstacles and
        creates the AUV.
        """
        self.path = RandomCurveThroughOrigin(rng=self.np_random)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        init_pos[0] += 2*(self.np_random.rand()-0.5)
        init_pos[1] += 2*(self.np_random.rand()-0.5)
        init_angle += 0.1*(self.np_random.rand()-0.5)
        self.vessel = AUV2D(self.config["t_step"],
                            np.hstack([init_pos, init_angle]))
        self.last_action = np.array([0, 0])

    def reset(self):
        """
        Resets the environment by reseeding and calling self.generate.

        Returns
        -------
        obs : np.array
            The initial observation of the environment.
        """
        self.vessel = None
        self.path = None
        self.obstacles = []
        self.reward = 0
        self.path_prog = 0

        if self.np_random is None:
            self.seed()

        self.generate()

        return self.observe()

    def observe(self):
        """
        Generates the observation of the environment.
        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position].

        Returns
        -------
        obs : np.array
            [
            surge velocity,
            sway velocity,
            heading error,
            distance to goal,
            propeller_input,
            rudder_positione,
            self.nsectors*[closeness to closest obstacle in sector]
            ]
            All observations are between -1 and 1.
        """
        la_dist = self.config["la_dist"]

        path_direction = self.path.get_direction(self.path_prog)
        target_heading = self.path.get_direction(
            self.path_prog + la_dist)

        heading_error = float(geom.princip(target_heading
                                           - self.vessel.heading))

        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([self.path(self.path_prog)
                       - self.vessel.position, 0]))[1]

        obs = np.zeros((self.nstates,))

        obs[0] = np.clip(self.vessel.velocity[0]
                         /self.vessel.max_speed, 0, 1)
        obs[1] = np.clip(self.vessel.velocity[1] / 0.26, -1, 1)
        obs[2] = np.clip(self.vessel.yawrate / 0.55, -1, 1)
        obs[3] = np.clip(heading_error / np.pi, -1, 1)
        obs[4] = np.clip(cross_track_error / la_dist, -1, 1)
        obs[5] = np.clip(self.last_action[0], 0, 1)
        obs[6] = np.clip(self.last_action[1], -1, 1)

        return obs


    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass