import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


class Critic(nn.Module):
    """Returns the action value function of a given observation, action pair"""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 0,
        action_dim: int = 16,
        q_network: DictConfig = {},
    ):
        super(Critic, self).__init__()
        q_network_cfg = OmegaConf.to_container(q_network, resolve=True)
        q_network_cfg.update({"input_dim": state_dim + goal_dim + action_dim})
        self.Q = hydra.utils.instantiate(q_network_cfg)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Performs a forward pass to obtain an action value function prediction"""
        # Case for batch size == 1
        if len(action.shape) == 2 and action.shape[0] == 1 and len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        q_input = torch.cat((obs, action), dim=-1)
        return self.Q(q_input)


class D2RLQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        with_dropout: bool = False,
        dropout_p: float = 0.3,
        num_layers: int = 2,
        last_layer_activation: str = "Identity",
        init_w: float = 1e-3,
    ):
        super(D2RLQNetwork, self).__init__()
        aux_dim = input_dim + hidden_dim
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(aux_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
        self.with_dropout = with_dropout
        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        self.last_layer_activation = getattr(nn, last_layer_activation)()

    def get_last_hidden_state(self, q_input):
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](q_input))
        for i in range(1, num_layers):
            x = torch.cat([x, q_input], dim=-1)
            x = F.silu(self.fc_layers[i](x))
        return x

    def forward(self, q_input):
        x = self.get_last_hidden_state(q_input)
        if self.with_dropout:
            x = self.dropout(x)
        return self.last_layer_activation((self.out(x)))


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        last_layer_activation: str = "Identity",
        init_w: float = 1e-3,
    ):
        super(MLPQNetwork, self).__init__()
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
        self.last_layer_activation = getattr(nn, last_layer_activation)()

    def forward(self, q_input):
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](q_input))
        for i in range(1, num_layers):
            x = F.silu(self.fc_layers[i](x))
        return self.last_layer_activation((self.out(x)))


class DenseNetQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        last_layer_activation: str = "Identity",
        init_w: float = 1e-3,
    ):
        super(DenseNetQNetwork, self).__init__()
        self.fc_layers = []
        fc_in_features = input_dim
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(fc_in_features, hidden_dim))
            fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(fc_in_features, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
        self.last_layer_activation = getattr(nn, last_layer_activation)()

    def forward(self, fc_input):
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return self.last_layer_activation((self.out(fc_input)))
