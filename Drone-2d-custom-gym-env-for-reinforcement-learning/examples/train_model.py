from pathlib import Path

import gym
from stable_baselines3 import PPO, SAC

# Need it to register the custom environment
import drone_2d_custom_gym_env

MODEL_DIR = Path(__file__).resolve().parent / "ppo_agents"
MODEL_DIR.mkdir(exist_ok=True)

# Select the algorithm to train: ppo or sac
ALGO = "ppo"

# Select the environment case to train on: 1 or 2
CASE_ID = 1

TIMESTEPS = 180000

CASES = {
    1: {"initial_throw": True, "initial_force": 5000, "initial_rotation_force": 600},
    2: {"initial_throw": True, "initial_force": 25000, "initial_rotation_force": 3000},
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
        n_fall_steps=10,
        change_target=True,
        initial_throw=case["initial_throw"],
        initial_force=case["initial_force"],
        initial_rotation_force=case["initial_rotation_force"],
    )


if __name__ == "__main__":
    env = make_env(CASE_ID)
    try:
        if ALGO == "ppo":
            model = PPO("MlpPolicy", env, verbose=1)
        elif ALGO == "sac":
            model = SAC("MlpPolicy", env, verbose=1)
        else:
            raise ValueError(f"Unsupported algorithm: {ALGO}")

        model.learn(total_timesteps=TIMESTEPS)
        model.save(str(MODEL_DIR / f"{ALGO}_agent_case{CASE_ID}"))
    finally:
        env.close()
