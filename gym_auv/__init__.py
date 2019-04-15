from gym.envs.registration import register

CONFIG = {
    "reward_ds": 1,
    "reward_closeness": -0.5,
    "reward_speed_error": -0.08,
    "reward_collision": -0.5,
    "nobstacles": 20,
    "obst_detection_range": 40,
    "obst_reward_range": 15,
    "t_step": 0.1,
    "cruise_speed": 1.4,
    "goal_dist": 400,
    "reward_goal": 0,
    "reward_rudderchange": 0,
    "min_reward": -500
}

register(
    id='AUV-v0',
    entry_point='gym_auv.envs:AUVEnv',
    kwargs={'env_config': CONFIG}
)
