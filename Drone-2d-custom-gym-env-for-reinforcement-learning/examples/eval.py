from stable_baselines3 import PPO
import gym
import time
import sys
from pathlib import Path

# Need it to register the custom environment
import drone_2d_custom_gym_env

MODEL_DIR = Path(__file__).resolve().parent / "ppo_agents"
CASE_ID = 1

continuous_mode = True #if True, after completing one episode the next one will start automatically
random_action = False #if True, the agent will take actions randomly

render_sim = True #if True, a graphic is generated

env = gym.make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
            shade_distance=70, n_steps=500, n_fall_steps=5, change_target=True,
            initial_throw=True, initial_force=5000, initial_rotation_force=600,wind=None,wind_magnitude=100)

"""
The example agent used here was originally trained with Python 3.7
For this reason, it is not compatible with Python version >= 3.8
Agent has been adapted to run in the newer version of Python,
but because of this, you cannot easily resume their training.
If you are interested in resuming learning, please use Python 3.7.
"""
if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    model_path = MODEL_DIR / f"ppo_agent_case{CASE_ID}"
else:
    model_path = MODEL_DIR / "ppo_agent_python3.7"

if not model_path.with_suffix(".zip").exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path.with_suffix('.zip')}\n"
        "Train first with train_model.py or update CASE_ID in eval.py."
    )

model = PPO.load(str(model_path), env=env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs = env.reset()

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        if done is True:
            if continuous_mode is True:
                obs = env.reset()
            else:
                break

finally:
    env.close()
