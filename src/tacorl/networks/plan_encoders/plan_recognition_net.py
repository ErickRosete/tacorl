from typing import Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.omegaconf import OmegaConf
from torch.distributions import Independent, Normal

from tacorl.utils.gym_utils import make_env
from tacorl.utils.test import get_simulated_trajectory


class PlanRecognitionNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_plan_dim: int,
        birnn_dropout_p: float,
        min_std: float,
        hidden_dim: int = 2048,
    ):
        super(PlanRecognitionNetwork, self).__init__()
        self.latent_plan_dim = latent_plan_dim
        self.min_std = min_std
        self.state_dim = state_dim
        self.birnn_model = nn.RNN(
            input_size=self.state_dim,
            hidden_size=hidden_dim,
            nonlinearity="relu",
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=birnn_dropout_p,
        )  # shape: [N, seq_len, 64+8]
        self.mean_fc = nn.Linear(
            in_features=2 * hidden_dim, out_features=self.latent_plan_dim
        )  # shape: [N, seq_len, 4096]
        self.variance_fc = nn.Linear(
            in_features=2 * hidden_dim, out_features=self.latent_plan_dim
        )  # shape: [N, seq_len, 4096]

    def forward(
        self, perceptual_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, hn = self.birnn_model(perceptual_emb)
        x = x[:, -1]  # we just need only last unit output
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        return mean, std  # shape: [N, 256]

    def __call__(self, *args, **kwargs):
        mean, std = super().__call__(*args, **kwargs)
        pr_dist = Independent(Normal(mean, std), 1)
        return pr_dist


@hydra.main(config_path="../../../config", config_name="test/networks_test")
def main(cfg):
    # Instantiate actor critic
    actor_critic_cfg = OmegaConf.to_container(cfg.actor_critic, resolve=True)
    actor_critic_cfg["env"] = make_env(cfg.env)
    actor_critic = hydra.utils.instantiate(actor_critic_cfg)
    actor = actor_critic.actor

    # Overwrite plan recognition params
    plan_recognition_cfg = OmegaConf.to_container(cfg.plan_recognition, resolve=True)
    plan_recognition_cfg["state_dim"] = actor.state_dim
    plan_recognition = hydra.utils.instantiate(plan_recognition_cfg)

    # Simulate forward pass
    sim_input = get_simulated_trajectory(**actor_critic.env_info)
    emb_input = actor.get_emb_representation(sim_input, cat_output=False)
    pr_dist = plan_recognition(emb_input)
    latent_plan = pr_dist.rsample()
    print(latent_plan.shape)


if __name__ == "__main__":
    main()
