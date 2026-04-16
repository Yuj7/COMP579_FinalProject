import torch
from torch import optim
import numpy as np
from ppo import PolicyNetwork, ValueNetwork
import gym
from torch.utils.data import random_split
from utils import Logger

class PPO():

    NORM_EPS = 1e-10

    def __init__(
            self,
            training_env : gym.Env,
            eval_env : gym.Env,
            policy_neurons_size : list[int],
            value_neurons_size : list[int],
            lr_policy : float,
            lr_value : float,
            gamma : float,
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
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr_policy)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr_value)
        self.value_criterion = torch.nn.MSELoss()

        self.gamma = gamma
        self.time_steps_per_batch = time_steps_per_batch
        self.epochs = epochs
        self.epsilon = epsilon
        self.mini_batch_size = mini_batch_size
        self.nb_eval_episodes = nb_eval_episodes

    def compute_return(self, rewards : list[float]) -> list[float]:
        returns = np.zeros_like(rewards, dtype = np.float32)
        for i in reversed(range(len(rewards))):
            if i + 1 == len(rewards):
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + self.gamma * returns[i + 1]
        return returns.tolist()

    def compute_advantage(self, returns : torch.Tensor, states : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            advantages = returns - self.value(states).squeeze(1)
        norm_advantages = (advantages - advantages.mean())/(advantages.std() + PPO.NORM_EPS)
        return norm_advantages, advantages

    def collect_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_t = 0

        batch_states, batch_actions, batch_next_states, batch_log_probs, batch_returns = [], [], [], [], []

        with torch.no_grad():
            while current_t < self.time_steps_per_batch:
                ep_rewards = []
                done = False
                state = self.training_env.reset()
                
                while not done and current_t < self.time_steps_per_batch:
                    action, log_prob = self.policy.get_action(torch.tensor(state, dtype = torch.float32))
                    action = action.numpy()
                    log_prob = log_prob.numpy()
                    next_state, reward, done, _ = self.training_env.step(action)

                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_next_states.append(next_state)
                    batch_log_probs.append(log_prob)

                    state = next_state

                    ep_rewards.append(reward)
                    current_t += 1
                
                batch_returns.extend(self.compute_return(ep_rewards))
        batch_returns = torch.tensor(np.array(batch_returns), dtype = torch.float32)
        batch_states = torch.tensor(np.array(batch_states), dtype = torch.float32)
        batch_actions = torch.tensor(np.array(batch_actions), dtype = torch.float32)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype = torch.float32)

        return batch_returns, batch_states, batch_actions, batch_log_probs

    def get_log_prob(self, batch_states : torch.Tensor, batch_actions : torch.Tensor) -> torch.Tensor:
        mu = self.policy(batch_states)
        dist = torch.distributions.Normal(mu, torch.exp(self.policy.log_std))
        log_probs = dist.log_prob(batch_actions).sum(axis = -1)
        return log_probs
    
    def learn(self):
        batch_returns, batch_states, batch_actions, batch_log_probs = self.collect_batch()
        for _ in range(self.epochs):

            # Shuffle indices
            sample_idx = torch.randperm(batch_states.size(0))

            # Collect minibatches   
            mini_batches_states, mini_batches_actions, mini_batches_log_probs, mini_batches_returns = self.split_into_mini_batch(sample_idx, batch_states, batch_actions, batch_log_probs, batch_returns)

            for states, actions, returns, log_probs in zip(mini_batches_states, mini_batches_actions, mini_batches_returns, mini_batches_log_probs):
                mini_batch_advantages, raw_adv = self.compute_advantage(returns, states)
                current_log_probs = self.get_log_prob(states, actions)
                ratios = torch.exp(current_log_probs)/torch.exp(log_probs.sum(axis = -1))
                
                first_surrogate = ratios * mini_batch_advantages
                second_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * mini_batch_advantages

                actor_loss = (-torch.min(first_surrogate, second_surrogate)).mean()
                critic_loss = self.value_criterion(self.value(states).squeeze(), returns)

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                critic_loss.backward()
                self.value_optimizer.step()

                with torch.no_grad():
                    actor_loss = actor_loss.mean().item()
                    critic_loss = critic_loss.mean().item()
                    raw_adv = raw_adv.mean().item()
                    self.add_metrics_to_log(actor_loss, critic_loss, raw_adv)
    
    def split_into_mini_batch(
            self,
            shuffled_batch_idx : torch.Tensor,
            batch_states : torch.Tensor,
            batch_actions : torch.Tensor,
            batch_log_probs : torch.Tensor,
            batch_returns : torch.Tensor
    ) -> list[torch.Tensor]:
        
        mini_batches_states = []
        mini_batches_actions = []
        mini_batches_log_probs = []
        mini_batches_returns = []

        if self.time_steps_per_batch % self.mini_batch_size == 0:
            mini_batches_sizes = [self.mini_batch_size for _ in range(int(self.time_steps_per_batch / self.mini_batch_size))]
            mini_batches = random_split(shuffled_batch_idx, mini_batches_sizes)
        else:
            mini_batches_sizes = [self.mini_batch_size for _ in range(int(self.time_steps_per_batch // self.mini_batch_size))]
            mini_batches_sizes.append(int(self.time_steps_per_batch - self.time_steps_per_batch // self.mini_batch_size * self.mini_batch_size))
            mini_batches = random_split(shuffled_batch_idx, mini_batches_sizes)

        for batch in mini_batches:
            mini_batches_states.append(batch_states[batch])
            mini_batches_actions.append(batch_actions[batch])
            mini_batches_log_probs.append(batch_log_probs[batch])
            mini_batches_returns.append(batch_returns[batch])

        return mini_batches_states, mini_batches_actions, mini_batches_log_probs, mini_batches_returns
    
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
                    next_state, reward, done, _ = self.training_env.step(action)

                    state = next_state

                    ep_rewards += reward
                
                eval_returns.append(reward)
            
        return np.mean(eval_returns), np.std(eval_returns)
    
    def add_metrics_to_log(self, actor_loss : float, critic_loss : float, advantage : float) -> None:
        self.logger.log['actor_loss'].append(round(actor_loss, 1))
        self.logger.log['value_loss'].append(round(critic_loss, 1))
        self.logger.log['advantages'].append(round(advantage, 1))