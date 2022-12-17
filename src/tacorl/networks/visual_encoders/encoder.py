from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Independent, Normal

from tacorl.networks.visual_encoders.utils import ResidualStack, SpatialSoftArgmax
from tacorl.utils.networks import freeze_params

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class CustomEncoder(nn.Module):
    """Configurable implementation of an Encoder with optional MLPs"""

    def __init__(
        self,
        input_width: int = 128,
        input_height: int = 128,
        input_channels: int = 3,
        kernel_sizes: List[int] = [3, 3, 3],
        n_channels: List[int] = [16, 16, 16],
        strides: List[int] = [1, 1, 1],
        paddings: List[int] = [1, 1, 1],
        latent_dim: int = 256,
        hidden_sizes: List[int] = None,
        conv_normalization_type: str = "none",
        fc_normalization_type: str = "none",
        init_w: float = 1e-4,
        hidden_init: Callable = nn.init.xavier_uniform_,
        pool_type: str = "none",
        pool_sizes: List = None,
        pool_strides: List = None,
        pool_paddings: List = None,
        spectral_norm_conv: bool = False,
        spectral_norm_fc: bool = False,
        normalize_conv_activation: bool = False,
        dropout: bool = False,
        dropout_prob: float = 0.2,
        activation_function: str = "ReLU",
        vib: bool = False,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == len(n_channels) == len(strides) == len(paddings)
        assert conv_normalization_type in {"none", "batch", "layer"}
        assert fc_normalization_type in {"none", "batch", "layer"}
        assert pool_type in {"none", "max2d"}
        if pool_type == "max2d":
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self.act_fn = getattr(nn, activation_function)()
        self.hidden_sizes = hidden_sizes
        self.input_channels = input_channels
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.conv_input_length = (
            self.input_width * self.input_height * self.input_channels
        )
        self.pool_type = pool_type

        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.dropout2d_layer = nn.Dropout2d(p=self.dropout_prob)

        self.spectral_norm_conv = spectral_norm_conv
        self.spectral_norm_fc = spectral_norm_fc

        self.normalize_conv_activation = normalize_conv_activation

        if normalize_conv_activation:
            print("normalizing conv activation")

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
            zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
            if self.spectral_norm_conv and 0 < i < len(n_channels) - 1:
                conv = nn.utils.spectral_norm(conv)

            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == "max2d":
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )

        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == "batch":
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == "layer":
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != "none" and len(self.pool_layers) > i:
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = self.get_conv_output_size()

        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()
        fc_input_size = self.conv_output_flat_size
        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            if self.spectral_norm_fc and 0 < idx < len(hidden_sizes) - 1:
                fc_layer = nn.utils.spectral_norm(fc_layer)
            fc_input_size = hidden_size

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)

            if self.fc_normalization_type == "batch":
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == "layer":
                self.fc_norm_layers.append(nn.LayerNorm(hidden_size))

        self.vib = vib
        if self.vib:
            self.fc_mean = nn.Linear(in_features=fc_input_size, out_features=latent_dim)
            self.fc_mean.weight.data.uniform_(-init_w, init_w)
            self.fc_mean.bias.data.uniform_(-init_w, init_w)

            self.fc_log_std = nn.Linear(
                in_features=fc_input_size, out_features=latent_dim
            )
            self.fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.last_fc = nn.Linear(fc_input_size, self.latent_dim)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

    def get_conv_output_size(self):
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self.conv_forward(test_mat)
        return int(np.prod(test_mat.shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.vib:
            dist = self.get_dist(input)
            return dist.rsample()
        else:
            x = self.conv_forward(input)
            if self.normalize_conv_activation:
                x = x / (torch.norm(x) + 1e-9)
            x = self.fc_forward(x)
            return self.last_fc(x)

    def get_dist(self, x: torch.Tensor) -> Normal:
        if self.vib:
            x = self.conv_forward(x)
            if self.normalize_conv_activation:
                x = x / (torch.norm(x) + 1e-9)
            x = self.fc_forward(x)
            mean = self.fc_mean(x)
            mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
            log_std = self.fc_log_std(x)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = log_std.exp()
            return Independent(Normal(mean, std), 1)
        return None

    def conv_forward(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != "none":
                h = self.conv_norm_layers[i](h)
            if self.pool_type != "none" and len(self.pool_layers) > i:
                h = self.pool_layers[i](h)
            if self.dropout:
                h = self.dropout2d_layer(h)
            h = self.act_fn(h)
        return torch.flatten(h, start_dim=1)

    def fc_forward(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != "none":
                h = self.fc_norm_layers[i](h)
            if self.dropout:
                h = self.dropout_layer(h)
            h = self.act_fn(h)
        return h


class ResNetRLEncoder(nn.Module):
    def __init__(
        self,
        input_width: int = 64,
        input_height: int = 64,
        input_channels: int = 3,
        hidden_channels: int = 128,
        latent_dim: int = 32,
        activation_function: str = "ReLU",
        normalize_output: bool = False,
        residual_hidden_channels: int = 64,
        num_residual_blocks: int = 3,
        spectral_norm: bool = False,
        vib: bool = False,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.latent_dim = latent_dim
        self.normalize_output = normalize_output
        self.act_fn = getattr(nn, activation_function)()
        self.vib = vib

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=hidden_channels // 2,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if spectral_norm:
            print("Building encoder with spectral norm")
            # Don't use spectral norm on the first layer
            self.conv_2 = nn.utils.spectral_norm(self.conv_2)
            self.conv_3 = nn.utils.spectral_norm(self.conv_3)

        self.residual_stack = ResidualStack(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            residual_hidden_channels=residual_hidden_channels,
            spectral_norm=spectral_norm,
            activation_function=activation_function,
        )
        self.flatten_dim = self.get_conv_output_size()
        if self.vib:
            self.fc1 = nn.Linear(in_features=self.flatten_dim, out_features=latent_dim)
        else:
            self.fc_mean = nn.Linear(
                in_features=self.flatten_dim, out_features=latent_dim
            )
            self.fc_log_std = nn.Linear(
                in_features=self.flatten_dim, out_features=latent_dim
            )
        if self.normalize_output:
            self.layernorm = nn.LayerNorm(latent_dim)

    def get_conv_output_size(self):
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self.conv_forward(test_mat)
        return int(np.prod(test_mat.shape))

    def get_dist(self, x: torch.Tensor) -> Normal:
        if self.vib:
            x = self.conv_forward(x)
            mean = self.fc_mean(x)
            mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
            log_std = self.fc_log_std(x)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = log_std.exp()
            return Independent(Normal(mean, std), 1)
        return None

    def conv_forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.act_fn(x)
        x = self.conv_2(x)
        x = self.act_fn(x)
        x = self.conv_3(x)
        x = self.residual_stack(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vib:
            dist = self.get_dist(x)
            return dist.rsample()
        else:
            x = self.conv_forward(x)
            x = self.fc1(x)
            if self.normalize_output:
                x = self.layernorm(x)
            return x


class LMPVisionEncoder(nn.Module):
    """Reference: https://arxiv.org/pdf/2005.07648.pdf"""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        activation_function: str = "ReLU",
        dropout: float = 0.0,
        temperature: float = None,
        normalize_spatial_softmax: bool = False,
        normalize_output: bool = False,
        vib: bool = False,
    ):
        super(LMPVisionEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.normalize_output = normalize_output
        self.act_fn = getattr(nn, activation_function)()
        self.vib = vib
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            self.act_fn,
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            self.act_fn,
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            self.act_fn,
            SpatialSoftArgmax(temperature, normalize_spatial_softmax),
            nn.Flatten(),
        )
        if self.vib:
            self.fc_mean = nn.Linear(in_features=128, out_features=latent_dim)
            self.fc_log_std = nn.Linear(in_features=128, out_features=latent_dim)
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(in_features=128, out_features=hidden_dim),
                self.act_fn,
                nn.Dropout(dropout),
                nn.Linear(in_features=hidden_dim, out_features=latent_dim),
            )
        if self.normalize_output:
            self.layernorm = nn.LayerNorm(latent_dim)

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vib:
            dist = self.get_dist(x)
            return dist.rsample()
        else:
            x = self.model(x)
            x = self.fc_layers(x)
            if self.normalize_output:
                x = self.layernorm(x)
            return x

    def get_dist(self, x: torch.Tensor) -> Normal:
        x = self.model(x)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return Independent(Normal(mean, std), 1)


class DeepSpatialEncoder(nn.Module):
    """Convolutional encoder of pixels observations.
    https://arxiv.org/pdf/1509.06113.pdf
    """

    def __init__(
        self,
        input_channels: int = 3,
        temperature: float = None,
        normalize: bool = False,
        freeze_backbone: bool = False,
        activation_function: str = "ReLU",
    ):

        super().__init__()
        self.latent_dim = 32
        out_channels = [64, 32, 16]
        kernel_size = [7, 5, 5]
        self.act_fn = getattr(nn, activation_function)()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, out_channels[0], kernel_size[0], stride=2),
            nn.BatchNorm2d(out_channels[0]),
            self.act_fn,
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size[1], stride=1),
            nn.BatchNorm2d(out_channels[1]),
            self.act_fn,
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size[2], stride=1),
            nn.BatchNorm2d(out_channels[2]),
            self.act_fn,
            SpatialSoftArgmax(temperature, normalize),
        )

        if freeze_backbone:
            freeze_params(self.backbone)

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, obs):
        return self.conv_forward(obs)


class ResNet18(nn.Module):
    """Convolutional encoder of pixels observations.
    https://arxiv.org/pdf/1509.06113.pdf
    """

    def __init__(
        self,
        latent_dim: int = 32,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        backbone = models.resnet18(pretrained=pretrained)
        n_inputs = backbone.fc.in_features
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc_layers = nn.Sequential(nn.Linear(n_inputs, latent_dim))
        if freeze_backbone:
            freeze_params(self.backbone)

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, obs):
        x = self.conv_forward(obs)
        return self.fc_layers(x)


class R3MResNet(nn.Module):
    """Convolutional encoder of pixels observations.
    https://github.com/facebookresearch/r3m
    """

    def __init__(
        self,
        device: str,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        activation_function: str = "ReLU",
        resnet_model: str = "resnet18",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        from r3m import load_r3m

        self.act_fn = getattr(nn, activation_function)()
        self.latent_dim = latent_dim
        print("r3m device:", device)
        self.backbone = load_r3m(resnet_model, device=device).module
        n_inputs = self.backbone.outdim
        self.fc_layers = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, latent_dim),
        )
        # set all grads to false
        freeze_params(self.backbone)
        if not freeze_backbone:
            # finetune last layer
            for param in self.backbone.convnet.layer4.parameters():
                param.requires_grad = True

    def conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, obs):
        x = self.conv_forward(obs)
        return self.fc_layers(x)


# @hydra.main(config_path="../../../config", config_name="test/networks_test")
def main():
    encoder = R3MResNet().cuda()
    input = torch.randn((64, 3, 128, 128), device="cuda") * 255
    output = encoder(input)
    print(output.shape)


if __name__ == "__main__":
    main()
