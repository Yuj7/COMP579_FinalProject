from pathlib import Path

import gym
from stable_baselines3 import PPO, SAC

# Need it to register the custom environment
import drone_2d_custom_gym_env

MODEL_DIR = Path(__file__).resolve().parent / "ppo_agents"

CASES = {
    1: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600},
    2: {"initial_throw": True, "initial_force": 25000, "initial_rotation_force": 3000},
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
        n_fall_steps=10,
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


def run_episode(model, env, render = False, continuous = False):
    obs = env.reset()
    episode_reward = 0.0
    done = False

    while True:
        if render:
            env.render()

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += float(reward)

        if done:
            if continuous:
                obs = env.reset()
                done = False
            else:
                break

    return episode_reward


if __name__ == "__main__":
    for algo in ("ppo", "sac"):
        for case_id in (1, 2):
            env = make_env(case_id, render_sim=False)
            try:
                model = load_model(algo, case_id, env)
                total_reward = run_episode(model, env, render=False)
                print(f"algo={algo} case={case_id} total_reward={total_reward:.2f}")
            finally:
                env.close()
