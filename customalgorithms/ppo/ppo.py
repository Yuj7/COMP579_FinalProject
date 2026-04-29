import torch
from torch import optim
import numpy as np
from .networks import PolicyNetwork, ValueNetwork
import gym
from utils import Logger, RolloutBatch
from eval import Evaluation

class CustomPPO():

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
            entropy_coeff : float,
            mini_batch_size : int,
            eval_freq : int,
            nb_eval_episodes : int,
            save_file_name : str,
            logger : Logger.PPO | None = None
    ) -> None:
        self.log = False
        if logger is not None:
            self.log = True
            
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
        self.entropy_coeff = entropy_coeff
        self.rollout_batch = RolloutBatch(self.time_steps_per_batch, self.training_env)

        self.eval_freq = eval_freq
        self.nb_eval_episodes = nb_eval_episodes
        self.total_steps = 0
        self.last_eval_step = 0
        self.eval_avg_return = []
        self.eval_ep_lengths = []
        self.eval_successes = []
        self.current_best = -np.inf
        
        self.save_file_name = save_file_name

        self._last_obs = self.training_env.reset()

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
        self.rollout_batch.reset()
        state = self._last_obs

        with torch.no_grad():
            ep_rewards = []
            done = False
            
            while current_t < self.time_steps_per_batch:
                action, raw_action, log_prob = self.policy.get_action(torch.tensor(state, dtype = torch.float32))
                action = action.numpy()
                raw_action = raw_action.numpy()
                log_prob = log_prob.numpy()
                next_state, reward, done, _ = self.training_env.step(action)

                self.rollout_batch.states[current_t, :] = torch.from_numpy(state)
                self.rollout_batch.actions[current_t, :] = torch.from_numpy(raw_action)
                self.rollout_batch.next_states[current_t, :] = torch.from_numpy(next_state)
                self.rollout_batch.log_probs[current_t] = torch.from_numpy(log_prob)
                self.rollout_batch.rewards[current_t] = reward
                self.rollout_batch.dones[current_t] = done

                state = next_state
                current_t += 1
                self.total_steps += 1

                if self.total_steps - self.last_eval_step >= self.eval_freq:
                    self.run_eval()
                    self.last_eval_step = self.total_steps

                ep_rewards.append(reward)
                self._last_obs = state

                if done:
                    state = self.training_env.reset()

            self.rollout_batch.values = self.value(self.rollout_batch.states).squeeze()   
            self.rollout_batch.next_values = self.value(self.rollout_batch.next_states).squeeze()

            norm_adv, adv, critic_target = self.compute_gae(self.rollout_batch.rewards, self.rollout_batch.values, self.rollout_batch.next_values, self.rollout_batch.dones)

            self.rollout_batch.norm_adv = norm_adv
            self.rollout_batch.adv = adv
            self.rollout_batch.critic_target = critic_target

    def get_log_prob(self, batch_states : torch.Tensor, batch_actions : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.policy(batch_states)
        dist = torch.distributions.Normal(mu, torch.exp(self.policy.log_std))
        entropy = dist.entropy().sum(-1).mean()
        projected_action = torch.tanh(batch_actions)
        log_probs = dist.log_prob(batch_actions) - torch.log(1 - projected_action.pow(2) + CustomPPO.EPS_TANH)
        return log_probs.sum(axis = -1, keepdim = True), entropy
    
    def learn(self):
        self.collect_batch()
        actor_loss_log = np.zeros(self.epochs)
        critic_loss_log = np.zeros(self.epochs)
        adv_log = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            actor_loss_per_epoch = []
            critic_loss_per_epoch = []
            adv_per_epoch = []

            for mb_states, mb_actions, mb_log_probs, mb_norm_adv, mb_adv, mb_critic_target  in self.rollout_batch.sample_mini_batches(mini_batch_size=self.mini_batch_size):

                current_log_probs, entropy = self.get_log_prob(mb_states, mb_actions)
                ratios = torch.exp(current_log_probs.squeeze() - mb_log_probs)
                
                first_surrogate = ratios * mb_norm_adv
                second_surrogate = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * mb_norm_adv

                actor_loss = (-torch.min(first_surrogate, second_surrogate)).mean() - self.entropy_coeff * entropy
                critic_loss = self.value_criterion(self.value(mb_states).squeeze(), mb_critic_target)

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.value_optimizer.step()

                if self.log:
                    with torch.no_grad():
                        actor_loss_per_epoch.append(actor_loss.mean().item())
                        critic_loss_per_epoch.append(critic_loss.mean().item())
                        adv_per_epoch.append(mb_adv.mean().item())

            if self.log:
                actor_loss_log[epoch] = np.mean(actor_loss_per_epoch)
                critic_loss_log[epoch] = np.mean(critic_loss_per_epoch)
                adv_log[epoch] = np.mean(adv_per_epoch)
        
        if self.log:
            self.add_metrics_to_log(np.mean(actor_loss_log), np.mean(critic_loss_log), np.mean(adv_log))
    
    def run_eval(self) -> tuple[np.ndarray, np.ndarray]:
        evaluation_phase = Evaluation(self.eval_env, self.nb_eval_episodes, self.policy.get_action)
        episodic_return, ep_length, successes = evaluation_phase.rollouts()
        self.eval_avg_return.append(episodic_return)
        self.eval_ep_lengths.append(ep_length)
        self.eval_successes.append(successes)

        if np.mean(episodic_return) > self.current_best:
            self.current_best = np.round(np.mean(episodic_return), 2)
            torch.save(self.policy.state_dict(), f'./models/{self.save_file_name}.pth')
        
        print(f'Current step: {self.total_steps}, avg return: {np.round(np.mean(episodic_return), 2)}, best avg return: {self.current_best}')
    
    def add_metrics_to_log(self, actor_loss : float, critic_loss : float, advantage : float) -> None:
        self.logger.log['actor_loss'].append(round(actor_loss, 1))
        self.logger.log['value_loss'].append(round(critic_loss, 1))
        self.logger.log['advantages'].append(round(advantage, 1))   