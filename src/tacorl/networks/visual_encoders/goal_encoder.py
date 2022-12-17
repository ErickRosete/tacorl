import torch
import torch.nn as nn


class VisualGoalEncoder(nn.Module):
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 32,
        hidden_size: int = 256,
        activation_function: str = "ReLU",
        last_layer_activation: str = "Identity",
        normalize_output: bool = False,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        self.act_fn = getattr(nn, activation_function)()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=out_features),
        )
        self.last_layer_activation = getattr(nn, last_layer_activation)()
        if self.normalize_output:
            self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        if self.normalize_output:
            x = self.layernorm(x)
        return self.last_layer_activation(x)
