import numpy as np
import torch

class PolicyNetwork(torch.nn.Module):
    def __init__(self, neurons_size : list[int], action_dim : int, state_dim : int) -> None:
        super(PolicyNetwork, self).__init__()
        self.action_dim = action_dim
        self.last_hidden_dim = neurons_size[-1]

        layers = [torch.nn.Linear(state_dim, neurons_size[0])]

        for i in range(len(neurons_size) - 1):
            layers.append(torch.nn.Linear(neurons_size[i], neurons_size[i+1]))

        layers.append(torch.nn.Linear(neurons_size[-1], action_dim))
        self.layers = torch.nn.ModuleList(layers)
        
        log_std = 0.5 * np.ones(action_dim, dtype = np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # self.apply(self.orthogonal_param_init)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        logit = x
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                logit = torch.relu(layer(logit))
            else:
                logit = layer(logit)
        return logit
    
    def orthogonal_param_init(self, model):
        if isinstance(model, torch.nn.Linear):
            with torch.no_grad():
                torch.nn.init.orthogonal_(model.weight)
                if model.out_features == self.action_dim and model.in_features == self.last_hidden_dim:
                    model.weight.div_(100)
                if model.bias is not None:
                    torch.nn.init.constant_(model.bias, 0)

    def get_action(self, state : torch.Tensor, deterministic : bool = False) ->  torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            mu = self.forward(state)
            return torch.tanh(mu)
        else:
            mu = self.forward(state)
            dist = torch.distributions.Normal(mu, torch.exp(self.log_std))
            action = dist.sample()
            projected_action = torch.tanh(action)
            log_prob = dist.log_prob(action) - torch.log(1 - projected_action.pow(2) + 1e-6)
        return projected_action, action, log_prob.sum(axis = -1, keepdim = True)

class ValueNetwork(torch.nn.Module):
    def __init__(self, neurons_size : list[int], state_dim : int) -> None:
        super(ValueNetwork, self).__init__()

        layers = [torch.nn.Linear(state_dim, neurons_size[0])]
                
        for i in range(len(neurons_size) - 1):
            layers.append(torch.nn.Linear(neurons_size[i], neurons_size[i+1]))

        layers.append(torch.nn.Linear(neurons_size[-1], 1))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        logit = x
        for idx, layer in enumerate(self.layers):
            if idx != len(self.layers) - 1:
                logit = torch.relu(layer(logit))
            else:
                logit = layer(logit)
        return logit