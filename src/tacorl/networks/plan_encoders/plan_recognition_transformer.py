import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tacorl.utils.distributions import TanhNormal


class PlanRecognitionTransformersNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_plan_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        encoder_hidden_size: int = 2048,
        fc_hidden_size: int = 4096,
        encoder_normalize: bool = False,
        positional_normalize: bool = False,
        position_embedding: bool = True,
        max_position_embeddings: int = 16,
        dropout_p: float = 0.01,
        min_std: float = 0.0001,
    ):

        super().__init__()
        self.state_dim = state_dim
        self.latent_plan_dim = latent_plan_dim
        self.padding = False
        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.encoder_normalize = encoder_normalize
        self.positional_normalize = positional_normalize
        self.min_std = min_std
        mod = self.state_dim % num_heads
        if mod != 0:
            print(f"Padding for Num of Heads : {num_heads}")
            self.padding = True
            self.pad = num_heads - mod
            self.state_dim += self.pad
        if position_embedding:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, self.state_dim
            )
        else:
            self.positional_encoder = PositionalEncoding(
                self.state_dim
            )  # TODO: with max window_size
        encoder_layer = nn.TransformerEncoderLayer(
            self.state_dim,
            num_heads,
            dim_feedforward=encoder_hidden_size,
            dropout=dropout_p,
        )
        encoder_norm = nn.LayerNorm(self.state_dim) if encoder_normalize else None
        self.layernorm = nn.LayerNorm(self.state_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.fc = nn.Linear(in_features=self.state_dim, out_features=fc_hidden_size)
        self.mean_fc = nn.Linear(
            in_features=fc_hidden_size, out_features=self.latent_plan_dim
        )
        self.variance_fc = nn.Linear(
            in_features=fc_hidden_size, out_features=self.latent_plan_dim
        )

    def forward(self, perceptual_emb: torch.Tensor) -> TanhNormal:
        batch_size, seq_len = perceptual_emb.shape[0], perceptual_emb.shape[1]
        perceptual_emb = (
            torch.cat(
                [
                    perceptual_emb,
                    torch.zeros((batch_size, seq_len, self.pad)).to(
                        perceptual_emb.device
                    ),
                ],
                dim=-1,
            )
            if self.padding
            else perceptual_emb
        )
        if self.position_embedding:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=perceptual_emb.device
            ).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            x = perceptual_emb + position_embeddings
            x = x.permute(1, 0, 2)
        else:
            # padd the perceptual embeddig
            x = self.positional_encoder(perceptual_emb.permute(1, 0, 2))  # [s, b, emb]
        if self.positional_normalize:
            x = self.layernorm(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x.permute(1, 0, 2))
        x = torch.mean(x, dim=1)  # gather all the sequence info
        mean = self.mean_fc(x)
        var = self.variance_fc(x)
        std = F.softplus(var) + self.min_std
        pr_dist = TanhNormal(mean, std)
        return pr_dist


class PositionalEncoding(nn.Module):
    """Implementation from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = (
            torch.cos(position * div_term)
            if d_model % 2 == 0
            else torch.cos(position * div_term[:-1])
        )
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
