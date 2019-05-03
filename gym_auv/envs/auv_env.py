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

class AUVEnv(gym.Env):
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
            env_config should contain the following members:
            reward_ds
                The reward for progressing ds along the path in
                one timestep. reward += reward_ds*ds.
            reward_closeness
                The reward for the closest obstacle within each
                sector. reward += reward_closeness*closeness.
            reward_surge_error
                The reward for going faster than the cruise_speed.
                reward += reward_surge_error*speeding_error
            reward_cross_track_error
                The reward for the cross track error at each timestep.
                reward += reward_cross_track_error*cross_track_error
            reward_collision
                The reward for colliding with an obstacle.
                reward += reward_collisions
            nobstacles
                The number of obstacles.
            los_dist
                The line of sight distance.
            obst_range
                The obstacle detection range.
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

        self.action_space = gym.spaces.Box(low=np.array([-1]*nactions),
                                           high=np.array([+1]*nactions),
                                           dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*nobservations),
            high=np.array([-1]*nobservations),
            dtype=np.float32)

    def step(self, action):
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

        if (self.reward < -300
            or abs(self.path_prog - self.path.length) < 1
            or dist_to_endpoint < 5):
            done = True

        return done, step_reward

    def generate(self):

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
        los_dist = self.config["los_dist"]

        path_direction = self.path.get_direction(self.path_prog)
        target_heading = self.path.get_direction(
            self.path_prog + los_dist)

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
        obs[4] = np.clip(cross_track_error / los_dist, -1, 1)
        obs[5] = np.clip(self.last_action[0], 0, 1)
        obs[6] = np.clip(self.last_action[1], -1, 1)

        return obs


    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass
