from typing import Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from tacorl.utils.distributions import GumbelSoftmax, TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class Actor(nn.Module):
    """
    Policy network, for continous action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 0,
        action_dim: int = 16,
        policy: DictConfig = {},
        discrete_gripper: bool = False,
    ):

        super(Actor, self).__init__()

        # Hyperparameters
        self.discrete_gripper = discrete_gripper
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        policy_cfg = OmegaConf.to_container(policy, resolve=True)
        policy_cfg.update(
            {
                "input_dim": self.state_dim + self.goal_dim,
                "action_dim": self.action_dim,
                "discrete_gripper": self.discrete_gripper,
            }
        )
        self.policy = hydra.utils.instantiate(policy_cfg)

    def forward(
        self, state_emb: torch.Tensor, goal_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if goal_emb is not None:
            x = torch.cat([state_emb, goal_emb], dim=-1)
        else:
            x = state_emb
        return self.policy(x)

    def get_dist(
        self, state_emb: torch.Tensor, goal_emb: Optional[torch.Tensor] = None
    ):
        mean, std = self.forward(state_emb, goal_emb)
        return TanhNormal(mean, std)

    def get_actions(
        self,
        observation: dict,
        deterministic: bool = False,
        reparameterize: bool = False,
    ):
        if self.discrete_gripper:
            mean, std, grip_act_logits = self.forward(observation)

            if deterministic:
                actions = torch.tanh(mean)
                gripper_probs = F.softmax(grip_act_logits, dim=-1)
                gripper_action = torch.argmax(gripper_probs, dim=-1)
                gripper_action = gripper_action.unsqueeze(-1) * 2.0 - 1
                actions = torch.cat((actions, gripper_action), dim=-1)
                log_pi = torch.zeros_like(actions)
                return actions, log_pi

            tanh_normal = TanhNormal(mean, std)
            gripper_dist = GumbelSoftmax(temperature=0.5, logits=grip_act_logits)
            if reparameterize:
                actions, log_pi = tanh_normal.rsample_and_logprob()
                gripper_action = gripper_dist.rsample(hard=True)  # one-hot
                gripper_action = torch.argmax(gripper_action, dim=-1)
            else:
                actions, log_pi = tanh_normal.sample_and_logprob()
                gripper_action = gripper_dist.sample()

            gripper_log_pi = gripper_dist.log_prob(gripper_action)
            log_pi = log_pi + gripper_log_pi
            gripper_action = gripper_action.unsqueeze(-1) * 2.0 - 1
            actions = torch.cat((actions, gripper_action), dim=-1)
            return actions, log_pi  # (bs, action_dim), (bs, 1)
        else:
            mean, std = self.forward(observation)

            if deterministic:
                actions = torch.tanh(mean)
                log_pi = torch.zeros_like(actions)
                return actions, log_pi

            tanh_normal = TanhNormal(mean, std)
            if reparameterize:
                actions, log_pi = tanh_normal.rsample_and_logprob()
            else:
                actions, log_pi = tanh_normal.sample_and_logprob()
            return actions, log_pi

    def sample_n_with_log_prob(
        self,
        observation: dict,
        n_actions: int,
    ):
        if self.discrete_gripper:
            mean, std, grip_act_logits = self.forward(observation)
            tanh_normal = TanhNormal(mean, std)
            actions, z = tanh_normal.sample_n(n_actions, return_pre_tanh_value=True)
            log_pi = tanh_normal.log_prob(actions, pre_tanh_value=z)

            gripper_dist = GumbelSoftmax(temperature=0.5, logits=grip_act_logits)
            gripper_action = gripper_dist.sample((n_actions,))
            gripper_log_pi = gripper_dist.log_prob(gripper_action)

            gripper_action = gripper_action.unsqueeze(-1) * 2 - 1
            actions = torch.cat((actions, gripper_action), dim=-1)

            log_pi = log_pi + gripper_log_pi
            return actions, log_pi  # (n, bs, action_dim), (n, bs, 1)
        else:
            mean, std = self.forward(observation)
            tanh_normal = TanhNormal(mean, std)
            actions, z = tanh_normal.sample_n(n_actions, return_pre_tanh_value=True)
            log_pi = tanh_normal.log_prob(actions, pre_tanh_value=z)
            return actions, log_pi  # (n, bs, action_dim), (n, bs, 1)

    def log_prob(self, observations, actions):
        if self.discrete_gripper:
            cont_actions = actions[..., :-1]
            mean, std, grip_act_logits = self.forward(observations)
            tanh_normal = TanhNormal(mean, std)
            log_pi = tanh_normal.log_prob(value=cont_actions)

            gripper_dist = GumbelSoftmax(temperature=0.5, logits=grip_act_logits)
            gripper_actions = actions[..., -1] / 2 + 0.5
            gripper_log_pi = gripper_dist.log_prob(gripper_actions)
            log_pi += gripper_log_pi
            return log_pi
        else:
            mean, std = self.forward(observations)
            tanh_normal = TanhNormal(mean, std)
            log_pi = tanh_normal.log_prob(value=actions)
            return log_pi


class D2RLPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
        discrete_gripper: bool = False,
    ):
        super(D2RLPolicy, self).__init__()
        self.discrete_gripper = discrete_gripper
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cont_action_dim = action_dim
        if self.discrete_gripper:
            self.gripper_action = nn.Linear(hidden_dim, 2)  # open / close
            self.gripper_action.weight.data.uniform_(-init_w, init_w)
            self.gripper_action.bias.data.uniform_(-init_w, init_w)
            cont_action_dim -= 1

        aux_dim = input_dim + hidden_dim
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(aux_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(hidden_dim, cont_action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, cont_action_dim)
        # https://arxiv.org/pdf/2006.05990.pdf
        # recommends initializing the policy MLP with smaller weights in the last layer
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def get_last_hidden_state(self, policy_input):
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](policy_input))
        for i in range(1, num_layers):
            x = torch.cat([x, policy_input], dim=-1)
            x = F.silu(self.fc_layers[i](x))
        return x

    def forward(self, policy_input):
        x = self.get_last_hidden_state(policy_input)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        if self.discrete_gripper:
            grip_act_logits = self.gripper_action(x)
            return mean, std, grip_act_logits
        else:
            return mean, std


class MLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
        discrete_gripper: bool = False,
    ):
        super(MLPPolicy, self).__init__()
        self.discrete_gripper = discrete_gripper
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cont_action_dim = action_dim
        if self.discrete_gripper:
            self.gripper_action = nn.Linear(hidden_dim, 2)  # open / close
            self.gripper_action.weight.data.uniform_(-init_w, init_w)
            self.gripper_action.bias.data.uniform_(-init_w, init_w)
            cont_action_dim -= 1

        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(hidden_dim, cont_action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, cont_action_dim)
        # https://arxiv.org/pdf/2006.05990.pdf
        # recommends initializing the policy MLP with smaller weights in the last layer
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def get_last_hidden_state(self, policy_input):
        num_layers = len(self.fc_layers)
        state = F.silu(self.fc_layers[0](policy_input))
        for i in range(1, num_layers):
            state = F.silu(self.fc_layers[i](state))
        return state

    def forward(self, policy_input):
        x = self.get_last_hidden_state(policy_input)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        if self.discrete_gripper:
            grip_act_logits = self.gripper_action(x)
            return mean, std, grip_act_logits
        else:
            return mean, std


class DenseNetPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
        discrete_gripper: bool = False,
    ):
        super(DenseNetPolicy, self).__init__()
        self.discrete_gripper = discrete_gripper
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cont_action_dim = action_dim
        if self.discrete_gripper:
            self.gripper_action = nn.Linear(hidden_dim, 2)  # open / close
            self.gripper_action.weight.data.uniform_(-init_w, init_w)
            self.gripper_action.bias.data.uniform_(-init_w, init_w)
            cont_action_dim -= 1

        self.fc_layers = []
        self.fc_in_features = input_dim
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(self.fc_in_features, hidden_dim))
            self.fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(self.fc_in_features, cont_action_dim)
        self.fc_log_std = nn.Linear(self.fc_in_features, cont_action_dim)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def get_last_hidden_state(self, fc_input):
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return fc_input

    def forward(self, policy_input):
        x = self.get_last_hidden_state(policy_input)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std
