import numpy as np
import torch
import gym

class Evaluation():
    def __init__(self, env : gym.Env, nb_eval_episodes : int, policy) -> None:
        self.env = env
        self.nb_eval_episodes = nb_eval_episodes
        self.policy = policy

    def rollouts(self) -> tuple[np.ndarray, np.ndarray]:
        eval_returns = []
        successes = []
        ep_length = []
        with torch.no_grad():
            for ep in range(self.nb_eval_episodes):
                state = self.env.reset()
                done = False
                ep_rewards = 0
                timestep = 0
                ep_success = False

                while not done:
                    action = self.policy.get_action(torch.tensor(state, dtype = torch.float32), deterministic = True)
                    action = action.numpy()
                    next_state, reward, done, info = self.env.step(action)

                    state = next_state

                    ep_rewards += reward
                    timestep += 1
                    if done:
                        ep_success = info.get("is_success")
                
                eval_returns.append(ep_rewards)
                ep_length.append(timestep)
                successes.append(float(ep_success))
            
        return eval_returns, ep_length, successes