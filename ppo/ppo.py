import torch
from torch import optim
import numpy as np
from ppo import PolicyNetwork, ValueNetwork
import gym
from torch.utils.data import random_split
from utils import Logger
from enum import Enum

class PPO():

    NORM_EPS = 1e-10
    EPS_TANH = 1e-6

    def __init__(
            self,
            training_env : gym.Env,
            eval_env : gym.Env,
            policy_neurons_size : list[int],
            value_neurons_size : list[int],
            lr_policy : float,
            lr_value : float,
            gamma : float,
            lamda : float,
            time_steps_per_batch : int,
            epochs : int,
            epsilon : float,
            mini_batch_size : int,
            nb_eval_episodes : int,
            logger : Logger.PPO
    ) -> None:
        self.logger = logger
        self.training_env = training_env
        self.eval_env = eval_env
        self.action_dim = training_env.action_space._shape[0]
        self.state_dim = training_env.observation_space._shape[0]

        self.policy = PolicyNetwork(policy_neurons_size, self.action_dim, self.state_dim)
        self.value = ValueNetwork(value_neurons_size, self.state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr_policy, eps = 1e-5)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr_value, eps = 1e-5)
        self.value_criterion = torch.nn.MSELoss()

        self.gamma = gamma
        self.lamda = lamda
        self.time_steps_per_batch = time_steps_per_batch
        self.epochs = epochs
        self.epsilon = epsilon
        self.mini_batch_size = mini_batch_size
        self.nb_eval_episodes = nb_eval_episodes

    def compute_gae(self, rewards : torch.Tensor, values : torch.Tensor, next_values : torch.Tensor, dones : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = len(rewards)
        advantages = torch.zeros(T)
        last_gae_lam = 0
        
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_values[t] * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lamda * next_non_terminal * last_gae_lam

        norm_advantages = (advantages - advantages.mean()) / (advantages.std() + self.NORM_EPS)
        critic_targets = advantages + values
        return norm_advantages, advantages, critic_targets

    def collect_batch(self) -> tuple:
        current_t = 0

        batch_states, batch_actions, batch_next_states, batch_log_probs, batch_rewards, batch_dones = [], [], [], [], [], []

        with torch.no_grad():
            while current_t < self.time_steps_per_batch:
                ep_rewards = []
                done = False
                state = self.training_env.reset()
                
                while not done and current_t < self.time_steps_per_batch:
                    action, raw_action, log_prob = self.policy.get_action(torch.tensor(state, dtype = torch.float32))
                    action = action.numpy()
                    raw_action = raw_action.numpy()
                    log_prob = log_prob.numpy()
                    next_state, reward, done, _ = self.training_env.step(action)

                    batch_states.append(state)
                    batch_actions.append(raw_action)
                    batch_next_states.append(next_state)
                    batch_log_probs.append(log_prob)
                    batch_rewards.append(reward)
                    batch_dones.append(float(done))

                    state = next_state

                    ep_rewards.append(reward)
                    current_t += 1

            batch_states = torch.tensor(np.array(batch_states), dtype = torch.float32)
            batch_actions = torch.tensor(np.array(batch_actions), dtype = torch.float32)
            batch_next_states = torch.tensor(np.array(batch_next_states), dtype = torch.float32)
            batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype = torch.float32)
            batch_rewards = torch.tensor(np.array(batch_rewards), dtype = torch.float32)
            batch_dones = torch.tensor(np.array(batch_dones), dtype = torch.float32)


            batch_values = self.value(batch_states).squeeze()   
            batch_next_values = self.value(batch_next_states).squeeze()

            # print(f'Rewards mean: {batch_rewards.mean()}, Values mean: {batch_values.mean()}')

        return batch_states, batch_actions, batch_rewards, batch_log_probs, batch_values, batch_next_values, batch_dones

    def get_log_prob(self, batch_states : torch.Tensor, batch_actions : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.policy(batch_states)
        dist = torch.distributions.Normal(mu, torch.exp(self.policy.log_std))
        entropy = dist.entropy().sum(-1).mean()
        projected_action = torch.tanh(batch_actions)
        log_probs = dist.log_prob(batch_actions) - torch.log(1 - projected_action.pow(2) + PPO.EPS_TANH)
        return log_probs.sum(axis = -1, keepdim = True), entropy
    
    def learn(self):
        batch_states, batch_actions, batch_rewards, batch_log_probs, batch_values, batch_next_values, batch_dones = self.collect_batch()
        batch_norm_advantages, batch_advantages, batch_critic_targets = self.compute_gae(batch_rewards, batch_values, batch_next_values, batch_dones)
        for _ in range(self.epochs):

            # Shuffle indices
            sample_idx = torch.randperm(batch_states.size(0))

            # Collect minibatches   
            states, actions, log_probs, norm_gaes, gaes, critic_targets = \
                self.split_into_mini_batch(sample_idx, batch_states, batch_actions, batch_log_probs, batch_norm_advantages, batch_advantages, batch_critic_targets)

            for state, action, log_prob, norm_gae, gae, critic_target in zip(states, actions, log_probs, norm_gaes, gaes, critic_targets):
    
                current_log_probs, entropy = self.get_log_prob(state, action)
                ratios = torch.exp(current_log_probs.squeeze() - log_prob)
                
                # print(f'mean ratio: {ratios.mean()}, std ratio: {ratios.std()}')
                first_surrogate = ratios * norm_gae
                second_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * norm_gae

                actor_loss = (-torch.min(first_surrogate, second_surrogate)).mean() - 0.1 * entropy
                critic_loss = self.value_criterion(self.value(state).squeeze(), critic_target)

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.value_optimizer.step()

                with torch.no_grad():
                    actor_loss = actor_loss.mean().item()
                    critic_loss = critic_loss.mean().item()
                    gae = gae.mean().item()
                    self.add_metrics_to_log(actor_loss, critic_loss, gae)
    
    def split_into_mini_batch(
            self,
            shuffled_batch_idx : torch.Tensor,
            batch_states : torch.Tensor,
            batch_actions : torch.Tensor,
            batch_log_probs : torch.Tensor,
            batch_norm_advantages : torch.Tensor,
            batch_advantages : torch.Tensor,
            batch_critic_targets : torch.Tensor
    ) -> list[torch.Tensor]:
        
        states = []
        actions = []
        log_probs = []
        norm_gaes = []
        gaes = []
        critics_targets = []

        if self.time_steps_per_batch % self.mini_batch_size == 0:
            mini_batches_sizes = [self.mini_batch_size for _ in range(int(self.time_steps_per_batch / self.mini_batch_size))]
            mini_batches = random_split(shuffled_batch_idx, mini_batches_sizes)
        else:
            mini_batches_sizes = [self.mini_batch_size for _ in range(int(self.time_steps_per_batch // self.mini_batch_size))]
            mini_batches_sizes.append(int(self.time_steps_per_batch - self.time_steps_per_batch // self.mini_batch_size * self.mini_batch_size))
            mini_batches = random_split(shuffled_batch_idx, mini_batches_sizes)

        for batch in mini_batches:
            states.append(batch_states[batch])
            actions.append(batch_actions[batch])
            log_probs.append(batch_log_probs[batch])
            norm_gaes.append(batch_norm_advantages[batch])
            gaes.append(batch_advantages[batch])
            critics_targets.append(batch_critic_targets[batch])

        return states, actions, log_probs, norm_gaes, gaes, critics_targets
    
    def evaluation_phase(self) -> tuple[np.ndarray, np.ndarray]:
        eval_returns = []
        with torch.no_grad():
            for ep in range(self.nb_eval_episodes):
                state = self.eval_env.reset()
                done = False
                ep_rewards = 0
                while not done:
                    action = self.policy.get_action(torch.tensor(state, dtype = torch.float32), deterministic = True)
                    action = action.numpy()
                    next_state, reward, done, _ = self.eval_env.step(action)

                    state = next_state

                    ep_rewards += reward
                
                eval_returns.append(ep_rewards)
            
        return np.mean(eval_returns), np.std(eval_returns)
    
    def add_metrics_to_log(self, actor_loss : float, critic_loss : float, advantage : float) -> None:
        self.logger.log['actor_loss'].append(round(actor_loss, 1))
        self.logger.log['value_loss'].append(round(critic_loss, 1))
        self.logger.log['advantages'].append(round(advantage, 1))