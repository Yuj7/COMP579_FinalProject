import numpy as np
import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self, neurons_size : list[int], action_dim : int, state_dim : int) -> None:
        super(PolicyNetwork, self).__init__()

        layers = [torch.nn.Linear(state_dim, neurons_size[0])]
        n_layers = len(neurons_size)
        for layer in range(1, n_layers - 1):
            layers.append(torch.nn.Linear(neurons_size[layer - 1], neurons_size[layer]))

        layers.append(torch.nn.Linear(neurons_size[-1], action_dim))
        self.layers = torch.nn.ModuleList(layers)
        
        log_std = 0.5 * np.ones(action_dim, dtype = np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        logit = x
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                logit = torch.relu(layer(logit))
        return logit

    def get_action(self, state : torch.Tensor, deterministic : bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            mu = self.forward(state)
            return torch.tanh(mu)
        else:
            mu = self.forward(state)
            dist = torch.distributions.MultivariateNormal(mu, torch.exp(self.log_std))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            projected_action = torch.tanh(action)
        return projected_action, log_prob

class ValueNetwork(torch.nn.Module):
    def __init__(self, neurons_size : list[int], state_dim : int) -> None:
        super(PolicyNetwork, self).__init__()
        layers = [torch.nn.Linear(state_dim, neurons_size[0])]
        n_layers = len(neurons_size)
        for layer in range(1, n_layers - 1):
            layers.append(torch.nn.Linear(neurons_size[layer - 1], neurons_size[layer]))

        layers.append(torch.nn.Linear(neurons_size[-1], 1))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        logit = x
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                logit = torch.relu(layer(logit))
        return logit
