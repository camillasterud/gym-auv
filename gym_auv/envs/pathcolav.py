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

class PathColavEnv(gym.Env):
    """
    Creates an environment with a vessel and a path.
    Attributes
    ----------
    config : dict
        The configuration disctionary specifying rewards,
        look ahead distance, simulation timestep and desired cruise
        speed.
    nstates : int
        The number of state variables passed to the agent.
    vessel : gym_auv.objects.auv.AUV2D
        The AUV that is controlled by the agent.
    path : gym_auv.objects.path.RandomCurveThroughOrigin
        The path to be followed.
    np_random : np.random.RandomState
        Random number generator.
    reward : float
        The accumulated reward
    path_prog : float
        Progression along the path in terms of arc length covered.
    past_actions : np.array
        All actions that have been perfomed.
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
            la_dist
                The look ahead distance.
            t_step
                The timestep
            cruise_speed
                The desired cruising speed.
        """
        self.config = env_config
        self.nstates = 6
        self.nsectors = 16
        nobservations = self.nstates + self.nsectors
        self.vessel = None
        self.path = None
        self.obstacles = None

        self.np_random = None

        self.reward = 0
        self.path_prog = None
        self.past_actions = None
        self.past_obs = None
        self.t_step = None

        self.reset()

        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float32)
        low_obs = [-1]*nobservations
        high_obs = [1]*nobservations
        low_obs[self.nstates - 1] = -10000
        high_obs[self.nstates - 1] = 10000
        self.observation_space = gym.spaces.Box(
            low=np.array(low_obs),
            high=np.array(high_obs),
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
        self.past_actions = np.vstack([self.past_actions, action])
        self.vessel.step(action)

        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.append(self.path_prog, prog)

        obs = self.observe()
        self.past_obs = np.vstack([self.past_obs, obs])
        done, step_reward, info = self.step_reward()

        return obs, step_reward, done, info

    def step_reward(self):
        """
        Calculates the step_reward and decides whether the episode
        should be ended.

        Parameters
        ----------
        obs : np.array
            The observation of the environment.
        delta_path_prog : double
            How much the vessel has progressed along the path
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
        info = {"collision": False}
        progress = self.path_prog[-1] - self.path_prog[-2]
        max_prog = self.config["cruise_speed"]*self.t_step
        speed_error = ((linalg.norm(self.vessel.velocity)
                        - self.config["cruise_speed"])
                       /self.vessel.max_speed)
        step_reward += (np.clip(progress/max_prog, -1, 1)
                        *self.config["reward_ds"])
        step_reward += (abs(self.past_obs[-1, self.nstates - 1])
                        *self.config["reward_cross_track_error"])
        step_reward += (max(speed_error, 0)
                        *self.config["reward_speed_error"])

        dist_to_endpoint = linalg.norm(self.vessel.position
                                       - self.path.get_endpoint())

        if (self.reward < self.config["min_reward"]
                or abs(self.path_prog[-1] - self.path.length) < 2
                or dist_to_endpoint < 5):
            done = True

        for sector in range(self.nsectors):
            closeness = self.past_obs[-1, self.nstates + sector]
            dist = self.config["obst_detection_range"]*(1 - closeness)
            reward_range = self.config["obst_reward_range"]
            if closeness >= 1:
                step_reward = self.config["reward_collision"]
                info["collision"] = True
                if self.config["end_on_collision"]:
                    done = True
            elif dist < reward_range:
                step_reward += ((1 - dist/reward_range)**2
                                *self.config["reward_closeness"])

        self.reward += step_reward

        return done, step_reward, info

    def generate(self):
        """
        Sets up the environment. Generates the path and
        initialises the AUV.
        """
        nwaypoints = int(np.floor(9*self.np_random.rand() + 2))
        self.path = RandomCurveThroughOrigin(self.np_random,
                                             nwaypoints, 400)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle
                                  + 2*np.pi*(self.np_random.rand()-0.5))
        self.t_step = self.config["t_step"]
        self.vessel = AUV2D(self.t_step,
                            np.hstack([init_pos, init_angle]))
        self.path_prog = np.array([
            self.path.get_closest_arclength(self.vessel.position)])

        max_obst_dist = max(linalg.norm(self.path(0)),
                            linalg.norm(self.path.get_endpoint()))

        for _ in range(self.config["nobstacles"]):
            obst_dist = 0.85*max_obst_dist*self.np_random.rand()
            obst_ang = 2*np.pi*(self.np_random.rand()-0.5)
            position = (obst_dist*np.array(
                [np.cos(obst_ang), np.sin(obst_ang)]))
            radius = 15*(self.np_random.rand()+1)
            self.obstacles.append(StaticObstacle(position, radius))

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
        self.path_prog = None
        self.past_actions = np.array([[0, 0]])
        self.t_step = None

        if self.np_random is None:
            self.seed()

        self.generate()
        obs = self.observe()
        self.past_obs = np.array([obs])
        return obs

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
            yawrate,
            heading error,
            cross track error,
            propeller_input,
            rudder_position,
            ]
            All observations are between -1 and 1.
        """
        la_heading = self.path.get_direction(
            self.path_prog[-1] + self.config["la_dist"])
        heading_error_la = float(geom.princip(la_heading
                                              - self.vessel.heading))
        path_position = (self.path(self.path_prog[-1]
                                   + self.config["la_dist"])
                         - self.vessel.position)
        target_heading = np.arctan2(path_position[1], path_position[0])
        heading_error = float(geom.princip(target_heading
                                           - self.vessel.heading))
        path_direction = self.path.get_direction(self.path_prog[-1])
        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([self.path(self.path_prog[-1])
                       - self.vessel.position, 0]))[1]

        obs = np.zeros((self.nstates + self.nsectors,))

        obs[0] = np.clip(self.vessel.velocity[0]
                         /self.vessel.max_speed, -1, 1)
        obs[1] = np.clip(self.vessel.velocity[1]
                         /0.26, -1, 1)
        obs[2] = np.clip(self.vessel.yawrate/0.55, -1, 1)
        obs[3] = np.clip(heading_error_la/np.pi, -1, 1)
        obs[4] = np.clip(heading_error/np.pi, -1, 1)
        obs[5] = np.clip(cross_track_error/50, -10000, 10000)

        obst_range = self.config["obst_detection_range"]
        for obst in self.obstacles:
            distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                np.hstack([obst.position - self.vessel.position, 0]))
            dist = linalg.norm(distance_vec)
            if dist < obst_range + obst.radius + self.vessel.radius:
                ang = ((float(np.arctan2(
                    distance_vec[1], distance_vec[0]))
                        + np.pi) / (2*np.pi))
                closeness = 1 - np.clip((dist - self.vessel.radius
                                         - obst.radius)/obst_range,
                                        0, 1)
                isector = (self.nstates
                           + int(np.floor(ang*self.nsectors)))
                if isector == self.nstates + self.nsectors:
                    isector = self.nstates
                if obs[isector] < closeness:
                    obs[isector] = closeness

        return obs


    def seed(self, seed=5):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass