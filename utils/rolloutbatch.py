import torch 
import numpy as np
import gym

class RolloutBatch():
    def __init__(self, batch_size : int, env : gym.Env) -> None:
        self.batch_size = batch_size
        self.env = env
        self.reset()

    def reset(self) -> None:
        self.states = torch.zeros((self.batch_size, self.env.observation_space._shape[0]), dtype = torch.float32)
        self.actions = torch.zeros((self.batch_size, self.env.action_space._shape[0]), dtype = torch.float32)
        self.next_states = torch.zeros((self.batch_size, self.env.observation_space._shape[0]), dtype = torch.float32)
        self.log_probs = torch.zeros(self.batch_size, dtype = torch.float32)
        self.rewards = torch.zeros(self.batch_size, dtype = torch.float32)
        self.dones = torch.zeros(self.batch_size, dtype = torch.float32)
        self.values = None
        self.next_values = None

        self.norm_adv = None
        self.adv = None
        self.critic_target = None

    def sample_mini_batches(self, mini_batch_size : torch.Tensor):
        indices = torch.randperm(self.batch_size)

        for i in range(0, self.batch_size, mini_batch_size):
            mb_idx = indices[i : i + mini_batch_size]

            yield (
                self.states[mb_idx],
                self.actions[mb_idx],
                self.log_probs[mb_idx],
                self.norm_adv[mb_idx],
                self.adv[mb_idx],
                self.critic_target[mb_idx]
            )