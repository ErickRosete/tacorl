from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tacorl.utils.distributions import TanhNormal


class PlanRecognitionTanhNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_plan_dim: int = 16,
        birnn_dropout_p: float = 0.0,
        min_std: float = 0.0001,
        hidden_dim: int = 2048,
    ):
        super(PlanRecognitionTanhNetwork, self).__init__()
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
        pr_dist = TanhNormal(mean, std)
        return pr_dist
