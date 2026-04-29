from pathlib import Path
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from customalgorithms import CustomPPO, train_seed

import drone_2d_custom_gym_env
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
from enum import Enum
from utils import Logger
import torch
from datetime import datetime

class Algorithm(Enum):
    CustomPPO = 'CustomPPO'
    CustomSAC = 'CustomSAC'
    BaselinePPO = 'BaselinePPO'
    BaselineSAC = 'BaselineSAC'

# Select the algorithm to train
ALGO = Algorithm.BaselineSAC

# Select the environment case to train on
CASE_ID = 1

CASES = {
    1: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind":None,"wind_magnitude":100.0},
    2: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind":None,"wind_magnitude":100.0},
    3: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind": "Uniform","wind_magnitude":100.0},
    4: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind": "Uniform","wind_magnitude":100.0},
    5: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind": "Random","wind_magnitude":100.0},
    6: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind": "Random","wind_magnitude":100.0}
}

def make_env(case_id):
    case = CASES[case_id]
    return gym.make(
        "drone-2d-custom-v0",
        render_sim=False,
        render_path=True,
        render_shade=True,
        shade_distance=70,
        n_steps=500,
        n_fall_steps=5,
        change_target=True,
        initial_throw=case["initial_throw"],
        initial_force=case["initial_force"],
        initial_rotation_force=case["initial_rotation_force"],
        wind=case["wind"],
        wind_magnitude=case["wind_magnitude"],
        step_penalty = 0,
        goal_reward = 20.0,
        death_penalty = -10.0,
        success_radius = 25.0
    )

TIMESTEPS = 180000

# Hyperparameters for Custom PPO
EVAL_FREQ_STEPS = 2000
NB_EVAL_EP = 10

# Hyperparameters for Custom SAC
REPLAYBUFFER_CAP = int(1e6)
START_STEPS = 100
UPDATE_AFTER = 100
UPDATE_EVERY = 1
BATCH_SIZE = 256

if __name__ == "__main__":
    start_time = datetime.now()
    
    for seed in range(3):
        np.random.seed(seed)
        torch.manual_seed(seed)

        training_env = make_env(CASE_ID)
        eval_env = make_env(CASE_ID)
        logger = Logger()
        try:
            if ALGO == Algorithm.BaselinePPO:
                model = PPO("MlpPolicy", training_env, verbose = 0, ent_coef = 0.01, learning_rate = 1e-4)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path = f"./models/{ALGO.value}_case{CASE_ID}_Seed{seed}",
                    log_path = f"./logs/{ALGO.value}_case{CASE_ID}_Seed{seed}",
                    eval_freq = EVAL_FREQ_STEPS, 
                    n_eval_episodes = NB_EVAL_EP, 
                    deterministic = True,
                    render = False,
                    verbose = 1
                    )
                model.learn(total_timesteps = TIMESTEPS, callback = eval_callback)

            elif ALGO == Algorithm.BaselineSAC:
                model = SAC("MlpPolicy", training_env, verbose = 0)
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path = f"./models/{ALGO.value}_case{CASE_ID}_Seed{seed}",
                    log_path = f"./logs/{ALGO.value}_case{CASE_ID}_Seed{seed}",
                    eval_freq = EVAL_FREQ_STEPS, 
                    n_eval_episodes = NB_EVAL_EP, 
                    deterministic = True,
                    render = False,
                    verbose = 1
                    )
                model.learn(total_timesteps = TIMESTEPS, callback = eval_callback)

            elif ALGO == Algorithm.CustomPPO:
                current_steps = 0

                ppo_agent = CustomPPO(
                    training_env = training_env,
                    eval_env = eval_env,
                    policy_neurons_size = [256, 256],
                    value_neurons_size = [256, 256],
                    lr_policy = 1e-4,
                    lr_value = 1e-4,
                    gamma = 0.99,
                    lamda = 0.95,
                    time_steps_per_batch = 2048,
                    epochs = 10,
                    epsilon = 0.2,
                    entropy_coeff = 0.01,
                    mini_batch_size = 64,
                    save_file_name = f'{ALGO.value}_Case{CASE_ID}_Seed{seed}',
                    eval_freq = EVAL_FREQ_STEPS,
                    nb_eval_episodes = NB_EVAL_EP,
                )

                while current_steps < TIMESTEPS:
                    ppo_agent.learn()
                    current_steps += ppo_agent.time_steps_per_batch

                logger.log_to_file(ppo_agent.eval_avg_return, ppo_agent.eval_ep_lengths, ppo_agent.eval_successes, f'{ALGO.value}_Case{CASE_ID}_Seed{seed}')

            elif ALGO == Algorithm.CustomSAC:
                history = train_seed(
                    seed = seed,
                    case_id = CASE_ID,
                    algo = ALGO.value,
                    training_env = training_env,
                    eval_env = eval_env,
                    buffer_capacity = REPLAYBUFFER_CAP,
                    start_steps = START_STEPS,
                    update_after = UPDATE_AFTER,
                    batch_size = BATCH_SIZE,
                    update_every = UPDATE_EVERY,
                    total_steps = TIMESTEPS
                )
                logger.log_to_file(history['eval_returns'], history['eval_lengths'], history['eval_successes'], f'{ALGO.value}_Case{CASE_ID}_Seed{seed}')
            else:
                raise ValueError(f"Unsupported algorithm: {ALGO}")

        finally:
            end_time = datetime.now()
            print(end_time - start_time) 
            training_env.close()
            eval_env.close()
