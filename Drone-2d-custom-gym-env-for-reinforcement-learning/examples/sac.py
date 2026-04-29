from collections import deque
import random
from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import torch.nn.functional as F
import torch
# Run this at the very top of your script
torch.set_num_threads(8)

class ReplayBuffer:
    def __init__(self, capacity,state_dim,act_dim):
        self.capacity=capacity

        self.ptr=0 #pointer  
        self.size_now=0

        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        







    def reset(self):
      self.ptr=0
      self.size_now=0
      

    def size(self):
      return self.size_now

    def store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

       



        self.ptr = (self.ptr + 1) % self.capacity # use as cirular buffer
        self.size_now = min(self.size_now + 1, self.capacity) 

    def uniform_sample(self,batch_size):
      idx = np.random.randint(0, self.size_now, batch_size)

      return {
        'obs': self.states[idx],
        'act': self.actions[idx],
        'rew': self.rewards[idx],
        'obs2': self.next_states[idx],
        'done': self.dones[idx]
    }

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        super().__init__()
        # Input is obs_dim + act_dim because we evaluate the pair
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1) # Output is a single Q-value
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
    


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256], log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_sizes[1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], act_dim)

    def forward(self, obs,deterministic=False,with_logprob=True):
        #obtain mean and log std
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        #set up distribution using std and mean
        std=torch.exp(log_std)
        dist=torch.distributions.Normal(mean,std)
        if deterministic:
            u=mean
        else:
            u=dist.rsample() # reparameterization trick
        
        if with_logprob:
            logp_pi=dist.log_prob(u).sum(axis=-1)
            #using tanh u directly leads to numerical instability, use identiy to rewrite log(1-tanh(u)^2) as log(1-tanh(u)^2)=log(1-u^2) to avoid numerical instability
            logp_pi-=(2*(np.log(2)-u-F.softplus(-2*u))).sum(axis=-1)
        action=torch.tanh(u)
        return action, logp_pi

class SACAgent:
    def __init__(self,obs_dim,act_dim,gamma=0.99):
        self.gamma=gamma
        #actor
        self.actor=GaussianActor(obs_dim,act_dim)
        #Double Q'
        self.q1=QNetwork(obs_dim,act_dim)
        self.q2=QNetwork(obs_dim,act_dim)

        #Target Qs
        self.q1_target=copy.deepcopy(self.q1)
        self.q2_target=copy.deepcopy(self.q2)

        #target network frozen, no update until we do soft update
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        #entry coefficient
        self.log_alpha=torch.zeros(1, requires_grad=True) # log alpha is a learnable parameter
        self.alpha_optimizer=optim.Adam([self.log_alpha],lr=3e-4)
        self.target_entropy=-act_dim # target entropy is -|A|, where |A| is the action dimension

    def compute_critic_loss(self,data):
        #get data from replay buffer
        states=data['obs']
        actions=data['act']
        rewards=data['rew'].view(-1, 1)# reshape to (batch_size, 1),2nd dimension to be 1,b x 1
        next_states=data['obs2']
        dones=data['done'].view(-1, 1) # reshape to (batch_size, 1)

        
        #target Q value,obtain bellman_target
        with torch.no_grad():
            next_actions,next_logp=self.actor(next_states)
            if next_logp.dim() == 1:#is actor returns logp with shape (batch_size,) , reshape to (batch_size,1)
                next_logp = next_logp.view(-1, 1)
                
            q1_target=self.q1_target(next_states,next_actions)
            q2_target=self.q2_target(next_states,next_actions)
            
            min_q_target=torch.min(q1_target,q2_target)

            #entropy term
            alpha=self.log_alpha.exp()
            bellman_target=rewards+self.gamma*(1-dones)*(min_q_target-alpha*next_logp)
        #compute critic loss
        q1_current=self.q1(states,actions)
        q2_current=self.q2(states,actions)

        critic_loss=F.mse_loss(q1_current,bellman_target)+F.mse_loss(q2_current,bellman_target)

        return critic_loss
    
    def compute_actor_loss(self,data):
        states=data['obs']
        actions,logp=self.actor(states)

      

        q1_new=self.q1(states,actions)
        q2_new=self.q2(states,actions)

        min_q_new=torch.min(q1_new,q2_new)

        alpha=self.log_alpha.exp().detach()# detach to avoid backprop grad leak
        actor_loss=(alpha*logp-min_q_new).mean()

        return actor_loss
    
    def compute_alpha_loss(self,data,target_entropy):
        states=data['obs']
        with torch.no_grad():
            actions,logp=self.actor(states)

        alpha_loss=-(self.log_alpha*(logp+target_entropy).detach()).mean()

        return alpha_loss
    def soft_update(self,polyak=0.995):
        with torch.no_grad():
            #put together local and target networks
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1 - polyak) * param.data)