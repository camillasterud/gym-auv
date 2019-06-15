from gym.envs.registration import register

CONFIG_COLAV = {
    "reward_ds": 1,
    "reward_closeness": -0.5,
    "reward_speed_error": -0.08,
    "reward_collision": -0.5,
    "nobstacles": 20,
    "obst_detection_range": 40,
    "obst_reward_range": 15,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "goal_dist": 400,
    "reward_rudderchange": 0,
    "min_reward": -500
}

CONFIG_PATHFOLLOWING = {
    "reward_ds": 1,
    "reward_speed_error": -0.08,
    "reward_cross_track_error": -0.5,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "la_dist": 10,
    "goal_dist": 400,
    "reward_rudderchange": 0,
    "min_reward": -500
}

CONFIG_PATHCOLAV = {
    "reward_ds": 1,
    "reward_speed_error": -0.08,
    "reward_cross_track_error": -0.5,
    "reward_closeness": -0.5,
    "reward_collision": -0.5,
    "nobstacles": 20,
    "obst_detection_range": 40,
    "obst_reward_range": 15,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "la_dist": 10,
    "goal_dist": 400,
    "reward_rudderchange": 0,
    "min_reward": -500
}

register(
    id='COLAV-v0',
    entry_point='gym_auv.envs:ColavEnv',
    kwargs={'env_config': CONFIG_COLAV}
)

register(
    id='PathFollowing-v0',
    entry_point='gym_auv.envs:PathFollowingEnv',
    kwargs={'env_config': CONFIG_PATHFOLLOWING}
)

register(
    id='PathColav-v0',
    entry_point='gym_auv.envs:PathColavEnv',
    kwargs={'env_config': CONFIG_PATHCOLAV}
)
