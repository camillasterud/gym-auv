from gym.envs.registration import register

CONFIG = {
    "reward_ds": 0.8,
    "reward_closeness": -0.3,
    "reward_surge_error": -0.1,
    "reward_cross_track_error": -0.6,
    "reward_collision": -0.3,
    "nobstacles": 10,
    "los_dist": 75,
    "obst_range": 100,
    "t_step": 0.1,
    "cruise_speed": 1.5
}

register(
    id='AUV-v0',
    entry_point='gym_auv.envs:AUVEnv',
    kwargs={'env_config': CONFIG}
)
