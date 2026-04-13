import torch
from torch import optim
import numpy as np
from ppo import PolicyNetwork, ValueNetwork
import gym

class PPO():

    NORM_EPS = 1e-10

    def __init__(
            self,
            env : gym.Env,
            policy_neurons_size : list[int],
            value_neurons_size : list[int],
            lr_policy : float,
            lr_value : float,
            gamma : float,
            time_steps_per_batch : int,
            nb_updates_per_iter : int,
            epsilon : float
    ) -> None:
        
        self.env = env,
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        self.policy = PolicyNetwork(policy_neurons_size, self.action_dim, self.state_dim)
        self.value = ValueNetwork(value_neurons_size, self.state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr_policy)
        self.value_optimizer = optim.Adam(self.policy.parameters(), lr_value)
        self.value_criterion = torch.nn.MSELoss()

        self.gamma = gamma
        self.time_steps_per_batch = time_steps_per_batch
        self.nb_updates_per_iter = nb_updates_per_iter
        self.epsilon = epsilon
        
    def compute_return(rewards : np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            if i + 1 == rewards.shape[0]:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + 0.9 * returns[i + 1]
        return returns

    def compute_advantage(self, returns : torch.Tensor, states : torch.Tensor):
        advantages = returns - self.value(states)
        norm_advantages = (advantages - advantages.mean())/(advantages.std() + PPO.NORM_EPS)
        return norm_advantages

    def collect_batch(self):
        current_t = 0

        batch_states = list(self.time_steps_per_batch)
        batch_actions = list(self.time_steps_per_batch)
        batch_next_states = list(self.time_steps_per_batch)
        batch_log_probs = list(self.time_steps_per_batch)
        batch_returns = list(self.time_steps_per_batch)

        while current_t < self.time_steps_per_batch:
            ep_rewards = []
            done = False
            state = self.env.reset()
            while not done:
                state = torch.tensor(state)
                action, log_prob = self.policy.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                batch_states.append(state)
                batch_actions.append(action)
                batch_next_states.append(next_state)
                batch_log_probs.append(log_prob)

                state = next_state

                ep_rewards.append(reward)

            batch_returns.append(self.compute_return(ep_rewards))

        batch_returns = torch.tensor(np.array(batch_returns)).flatten()
        batch_states = torch.tensor(batch_states)
        batch_actions = torch.tensor(batch_actions)
        batch_log_probs = torch.tensor(batch_log_probs)


    def get_log_prob(self, batch_states : torch.Tensor, batch_actions : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.policy(batch_states)
        dist = torch.distributions.MultivariateNormal(mu, torch.exp(self.policy.log_std))
        log_probs = dist.log_prob(batch_actions)
        return log_probs
    
    def learn(self):
        for _ in range(self.nb_updates_per_iter):
            batch_states, batch_actions, batch_log_probs, batch_returns = self.collect_batch()

            batch_advantages = self.compute_advantage(batch_returns, batch_states)

            current_log_probs = self.get_log_prob(batch_states, batch_actions)

            ratios = torch.exp(batch_log_probs)/torch.exp(current_log_probs)

            first_surrogate = ratios * batch_advantages
            second_surrogate = torch.clamp(1 - self.epsilon, 1 + self.epsilon, ratios) * batch_advantages

            actor_loss = (-torch.min(first_surrogate, second_surrogate)).mean()
            critic_loss = self.value_criterion(self.value(batch_states).squeeze(), batch_returns)

            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()