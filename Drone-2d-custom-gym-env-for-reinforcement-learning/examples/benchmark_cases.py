from pathlib import Path
from collections import Counter

import gym
import numpy as np
from stable_baselines3 import PPO, SAC

# Need it to register the custom environment
import drone_2d_custom_gym_env

MODEL_DIR = Path(__file__).resolve().parent / "ppo_agents"
NUM_EPISODES = 100

CASES = {
    1: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind":None,"wind_magnitude":100},
    2: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind":None,"wind_magnitude":100},
    3: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind": "Uniform","wind_magnitude":100},
    4: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind": "Uniform","wind_magnitude":100},
    5: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600,"wind": "Random","wind_magnitude":100},
    6: {"initial_throw": True, "initial_force": 12000, "initial_rotation_force": 1500,"wind": "Random","wind_magnitude":100}
}

# Point these to the saved model files
def model_path(algo, case_id):
    return MODEL_DIR / f"{algo}_agent_case{case_id}"

def make_env(case_id, render_sim = False):
    case = CASES[case_id]
    return gym.make(
        "drone-2d-custom-v0",
        render_sim=render_sim,
        render_path=True,
        render_shade=True,
        shade_distance=70,
        n_steps=500,
        n_fall_steps=5,
        change_target=True,
        initial_throw=case["initial_throw"],
        initial_force=case["initial_force"],
        initial_rotation_force=case["initial_rotation_force"],
    )

def load_model(algo, case_id, env):
    if algo == "ppo":
        return PPO.load(str(model_path(algo, case_id)), env=env)
    if algo == "sac":
        return SAC.load(str(model_path(algo, case_id)), env=env)
    raise ValueError(f"Unsupported algorithm: {algo}")

def run_episode(model, env, render = False):
    obs = env.reset()
    episode_reward = 0.0
    episode_steps = 0

    while True:
        if render:
            env.render()

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += float(reward)
        episode_steps += 1

        if done:
            terminal_status = info.get("terminal_status")
            return episode_reward, episode_steps, terminal_status


if __name__ == "__main__":
    for algo in ("ppo", "sac"):
        for case_id in (1, 2):
            env = make_env(case_id, render_sim=False)
            try:
                model = load_model(algo, case_id, env)
                rewards = []
                steps = []
                terminal_statuses = []

                for _ in range(NUM_EPISODES):
                    total_reward, episode_steps, terminal_status = run_episode(model, env, render=False)
                    rewards.append(total_reward)
                    steps.append(episode_steps)
                    terminal_statuses.append(terminal_status)

                status_counts = Counter(terminal_statuses)
                success_rate = status_counts.get("success", 0) / NUM_EPISODES
                print(
                    f"algo={algo} case={case_id} episodes={NUM_EPISODES} "
                    f"mean_reward={np.mean(rewards):.2f} mean_steps={np.mean(steps):.1f} "
                    f"success_rate={success_rate:.2%} status_counts={dict(status_counts)}"
                )
            finally:
                env.close()
