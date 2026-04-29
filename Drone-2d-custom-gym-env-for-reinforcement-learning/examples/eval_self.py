import torch
import numpy as np
import gym
import sys
from pathlib import Path
import drone_2d_custom_gym_env

from sac import SACAgent  # adjust import to wherever your classes live

# --- CONFIG ---
MODEL_DIR = Path.cwd() / "ppo_agents"
CASE_ID = 1
ALGO = "self_sac"
continuous_mode = True
random_action = False
render_sim = True
# --------------

env = gym.make(
    'drone-2d-custom-v0',
    render_sim=render_sim,
    render_path=True,
    render_shade=True,
    shade_distance=70,
    n_steps=500,
    n_fall_steps=5,
    change_target=True,
    initial_throw=True,
    initial_force=5000,
    initial_rotation_force=600,
    wind=None,
    wind_magnitude=100
)

# --- LOAD MODEL ---
model_path = MODEL_DIR / f"{ALGO}_agent_case{CASE_ID}.pt"

if not model_path.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}\n"
        "Train first or update CASE_ID."
    )

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
checkpoint = torch.load(model_path)
agent.actor.load_state_dict(checkpoint['actor_state_dict'])
agent.actor.eval()
# ------------------

obs = env.reset()
if isinstance(obs, tuple): obs = obs[0]

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                action, _ = agent.actor(obs_tensor, deterministic=True)
                action = action.numpy()

        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc

        if done:
            print(f"Episode finished | Status: {info.get('terminal_status', 'unknown')}")
            if continuous_mode:
                obs = env.reset()
                if isinstance(obs, tuple): obs = obs[0]
            else:
                break

finally:
    env.close()