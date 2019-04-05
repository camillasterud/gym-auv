from gym.envs.registration import register

CONFIG = {
    "reward_ds": 0.9,
    "reward_closeness": -1.2,
    "reward_surge_error": -0.1,
    "reward_cross_track_error": -0.5,
    "reward_collision": -1.2,
    "nobstacles": 0,
    "los_dist": 25,
    "obst_range": 100,
    "t_step": 0.1,
    "cruise_speed": 1.5,
    "goal_dist": 400,
    "reward_goal": 10
}

register(
    id='AUV-v0',
    entry_point='gym_auv.envs:AUVEnv',
    kwargs={'env_config': CONFIG}
)
