import gym
from gym.utils import seeding
import numpy as np
import numpy.linalg as linalg

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin
from gym_auv.objects.obstacles import StaticObstacle

class AUVEnv(gym.Env):

    def __init__(self, env_config):
        self.config = env_config
        self.nstates = 4
        self.nsectors = 4
        nactions = 2
        nobservations = self.nstates + self.nsectors
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
        self.observation_space = gym.spaces.Box(low=np.array([-1]*nobservations),
                                                high=np.array([-1]*nobservations),
                                                dtype=np.float32)

    def step(self, action):
        self.last_action = action
        self.vessel.step(action)

        path_prog = self.path.get_closest_arclength(self.vessel.position)
        delta_path_prog, self.path_prog = path_prog - self.path_prog, path_prog

        obs = self.observe()
        done, step_reward = self.step_reward(action, obs, delta_path_prog)
        info = {}
        self.reward += step_reward

        return obs, step_reward, done, info

    def step_reward(self, action, obs, delta_path_prog):
        done = False
        step_reward = 0

        if not done and self.reward < -300:
            done = True

        if not done and abs(self.path_prog - self.path.length) < 1:
            done = True

        #for o in self.static_obstacles + self.dynamic_obstacles:
        #    if not done and linalg.norm(self.vessel.position - o.position) < self.vessel.radius + o.radius:
        #        done = True
        #        step_reward += self.config["reward_collision"]
        #        break

        if not done and linalg.norm(self.vessel.position - self.path.get_endpoint()) < 10:
            done = True

        if not done:
            step_reward += delta_path_prog*self.config["reward_ds"]

        for sector in range(self.nsectors):
            closeness = obs[self.nstates + sector]
            step_reward += self.config["reward_closeness"]*closeness**2

        if not done:
            surge_error = obs[0] - self.config["cruise_speed"]/self.vessel.max_speed
            cross_track_error = obs[2]

            step_reward += abs(cross_track_error)*self.config["reward_cross_track_error"]
            step_reward += max(0, -surge_error)*self.config["reward_surge_error"]

        return done, step_reward

    def generate(self):

        self.path = RandomCurveThroughOrigin(rng=self.np_random)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        init_pos[0] += 2*(self.np_random.rand()-0.5)
        init_pos[1] += 2*(self.np_random.rand()-0.5)
        init_angle += 0.1*(self.np_random.rand()-0.5)
        self.vessel = AUV2D(self.config["t_step"], np.hstack([init_pos, init_angle]))
        self.last_action = np.array([0, 0])

        for _ in range(self.config["nobstacles"]):
            position = (self.path(0.9*self.path.length*(self.np_random.rand() + 0.1))
                        + 100*(self.np_random.rand(2)-0.5))
            radius = 10*(self.np_random.rand()+0.5)
            self.obstacles.append(StaticObstacle(position, radius))

    def reset(self):
        self.vessel = None
        self.path = None
        if self.np_random is None:
            self.seed()
        self.obstacles = []

        self.reward = 0
        self.path_prog = 0

        self.generate()

    def observe(self):
        los_dist = self.config["los_dist"]
        obst_range = self.config["obst_range"]

        closest_point = self.path(self.path_prog)
        target_angle = self.path.get_direction(self.path_prog + los_dist)

        heading_error = float(geom.princip(target_angle - self.vessel.heading))
        cross_track_error = linalg.norm(closest_point - self.vessel.position)

        obs = np.zeros((self.nstates + self.nsectors,))

        obs[0] = np.clip(linalg.norm(self.vessel.velocity) / self.vessel.max_speed, 0, 1)
        obs[1] = np.clip(heading_error / np.pi, -1, 1)
        obs[2] = np.clip(cross_track_error / los_dist, -1, 1)
        obs[3] = np.clip(self.last_action[0], -1, 1)
        obs[4] = np.clip(self.last_action[1], 0, 1)

        for obst in self.obstacles:
            distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                np.hstack([obst.position - self.vessel.position, 0]))
            dist = np.linalg.norm(distance_vec)
            if dist < obst_range:
                ang = (float(geom.princip(
                    np.arctan2(distance_vec[1], distance_vec[0])))
                       + np.pi/2) / np.pi
                if 0 <= ang < 1:
                    closeness = 1 - np.clip((dist - self.vessel.radius - obst.radius)
                                            / obst_range, 0, 1)
                    isector = self.nstates + int(np.floor(ang*self.nsectors))
                    if obs[isector] < closeness:
                        obs[isector] = closeness

        return obs


    def seed(self, seed=5):
        self.np_random, _ = seeding.np_random(seed)

    def render(self):
        pass
