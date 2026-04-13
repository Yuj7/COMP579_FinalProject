from drone_2d_custom_gym_env.drone_2d_env import *
from gym.envs.registration import register

register(
    id='drone-2d-custom-v0',
    entry_point='drone_2d_custom_gym_env:Drone2dEnv',
    kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
            'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 5, 'change_target': False,
            'initial_throw': True, 'initial_force': 5000, 'initial_rotation_force': 600,
            'step_penalty': 0.1, 'goal_reward': 100.0, 'death_penalty': -100.0,
            'success_radius': 25.0,'wind': None,'wind_magnitude':100}
)

# register(
#     id='drone-2d-custom-v0',
#     entry_point='drone_2d_custom_gym_env:Drone2dEnv',
#     kwargs={'render_sim': False, 'render_path': True, 'render_shade': True,
#             'shade_distance': 75, 'n_steps': 500, 'n_fall_steps': 10, 'change_target': False,
#             'initial_throw': True, 'initial_force': 25000, 'initial_rotation_force': 3000}
# )
