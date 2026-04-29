from collections import deque
import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from .sac import SACAgent,ReplayBuffer

import torch.nn.functional as F
import torch
from eval import Evaluation

SEEDS = [1, 2, 3]

def train_seed(
        seed : int,
        case_id : int,
        algo,
        training_env,
        eval_env,
        buffer_capacity : int,
        start_steps : int,
        update_after : int,
        batch_size : int,
        update_every : int,
        total_steps : int,
        eval_freq : int,
        nb_eval_episodes : int
):
    
    training_env.reset()

    obs_dim = training_env.observation_space.shape[0]
    act_dim = training_env.action_space.shape[0]

    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
    buffer = ReplayBuffer(buffer_capacity, act_dim=(act_dim,), state_dim=(obs_dim,))

    history = {
        'steps': [],
        'rewards': [],
        'lengths': [],
        'successes': [],
        'eval_returns' : [],
        'eval_lengths' : [],
        'eval_successes' : []
    }

    curr_ep_reward = 0
    curr_ep_len = 0

    last_eval_step = 0
    current_best = -np.inf

    q_optimizer = optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=3e-4)
    pi_optimizer = optim.Adam(agent.actor.parameters(), lr=3e-4)
    alpha_optimizer = agent.alpha_optimizer

    obs = training_env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    for t in range(total_steps):
        # if t % 1000 == 0:
        #     print(f"[Seed {seed}] Step: {t} | Avg Reward: {np.mean(history['rewards']) if history['rewards'] else 0:.4f} | Avg Length: {np.mean(history['lengths']) if history['lengths'] else 0:.4f}")

        if t - last_eval_step >= eval_freq or t == total_steps -1:
            evaluation_phase = Evaluation(eval_env, nb_eval_episodes, agent.actor.forward)
            episodic_return, ep_length, successes = evaluation_phase.rollouts()

            history['eval_returns'].append(episodic_return)
            history['eval_lengths'].append(ep_length)
            history['eval_successes'].append(successes)
        
            if np.mean(episodic_return) > current_best:
                current_best = episodic_return
            
            print(f'Current step: {t}, avg return: {np.round(np.mean(episodic_return), 2)}, best avg return: {current_best}')

        if t < start_steps:
            action = training_env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                action, _ = agent.actor(obs_tensor)
                action = action.detach().numpy()
        
        step_result = training_env.step(action)
        if len(step_result) == 4:
            next_obs, reward, done, info = step_result
        else:
            next_obs, reward, term, trunc, info = step_result
            done = term or trunc

        is_timeout = (info.get('terminal_status') == "timeout")
        real_done = float(done and not is_timeout)

        buffer.store(obs, action, reward, next_obs, real_done)

        curr_ep_reward += reward
        curr_ep_len += 1

        if done:
            history['steps'].append(t)
            history['rewards'].append(curr_ep_reward)
            history['lengths'].append(curr_ep_len)

            is_success = (info.get('terminal_status') == "success")
            history['successes'].append(1.0 if is_success else 0.0)

            curr_ep_reward, curr_ep_len = 0, 0
            obs = training_env.reset()
            if isinstance(obs, tuple): obs = obs[0]
        else:
            obs = next_obs

        # 3. Update
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch_numpy = buffer.uniform_sample(batch_size)
                batch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch_numpy.items()}

                q_optimizer.zero_grad()
                loss_critic = agent.compute_critic_loss(batch)
                loss_critic.backward()
                q_optimizer.step()

                pi_optimizer.zero_grad()
                loss_pi = agent.compute_actor_loss(batch)
                loss_pi.backward()
                pi_optimizer.step()

                alpha_optimizer.zero_grad()
                agent.compute_alpha_loss(batch, target_entropy=-act_dim).backward()
                alpha_optimizer.step()

                agent.soft_update(polyak=0.995)

    # # --- SAVE MODEL ---
    # MODEL_DIR = Path.cwd() / "ppo_agents"
    # MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # torch.save({
    #     'actor_state_dict': agent.actor.state_dict(),
    #     'q1_state_dict': agent.q1.state_dict(),
    #     'q2_state_dict': agent.q2.state_dict(),
    # }, str(MODEL_DIR / f"{algo}_agent_case{case_id}_seed{seed}.pt"))

    # # --- SAVE HISTORY ---
    # history_path = MODEL_DIR / f"{algo}_history_case{case_id}_seed{seed}.json"
    # with open(history_path, "w") as f:
    #     json.dump(history, f)

    # print(f"[Seed {seed}] Done. Model and history saved.")
    # training_env.close()

    return history


# def train_all_seeds():
#     all_histories = {}

#     for seed in SEEDS:
#         print(f"\n{'='*50}")
#         print(f"  Training Seed {seed}")
#         print(f"{'='*50}\n")
#         history = train_seed(seed, CASE_ID, ALGO)
#         all_histories[seed] = history

#     # Save combined history for easy plotting
#     MODEL_DIR = Path.cwd() / "ppo_agents"
#     combined_path = MODEL_DIR / f"{ALGO}_history_case{CASE_ID}_all_seeds.json"
#     with open(combined_path, "w") as f:
#         json.dump({str(k): v for k, v in all_histories.items()}, f)

#     print(f"\nAll seeds done. Combined history saved to {combined_path}")
#     return all_histories


# Run:
