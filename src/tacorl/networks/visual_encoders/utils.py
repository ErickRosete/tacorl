import numpy as np
import torch
import torch.nn as nn


def identity(x):
    return x


def fanin_init(tensor: torch.Tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature: float = None, normalize: bool = False):
        """
        Reference:
        https://github.com/gorosgobe/dsae-torch/blob/8db2e7ac55748995b23fe8cc4c4cdb76ac22c8f8/dsae.py#L27
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalize: Should spatial features be normalized to range [-1, 1]
        """
        super().__init__()
        self.temperature = (
            nn.Parameter(torch.ones(1))
            if temperature is None
            else torch.tensor([temperature])
        )
        self.normalize = normalize

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(
            x.view(n * c, h * w) / self.temperature, dim=1
        )
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = self.get_image_coordinates(
            h, w, normalize=self.normalize, device=x.device
        )
        # size (H, W, 2)
        image_coordinates = torch.cat(
            (image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1
        )

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        bs, c = out.shape[:2]
        return out.view(bs, c * 2)  # (N, C, 2) -> (N, 2*C)

    @staticmethod
    def get_image_coordinates(h, w, normalize, device="cuda"):
        x_range = torch.arange(w, dtype=torch.float32, device=device)
        y_range = torch.arange(h, dtype=torch.float32, device=device)
        if normalize:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        residual_hidden_channels: int = 64,
        spectral_norm: bool = False,
        activation_function: str = "ReLU",
    ):

        super(ResidualBlock, self).__init__()
        self.act_fn = getattr(nn, activation_function)()

        if spectral_norm:
            self._block = nn.Sequential(
                self.act_fn,
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=residual_hidden_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                ),
                self.act_fn,
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=residual_hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    )
                ),
            )
        else:
            self._block = nn.Sequential(
                self.act_fn,
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=residual_hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                self.act_fn,
                nn.Conv2d(
                    in_channels=residual_hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        num_residual_blocks: int = 3,
        residual_hidden_channels: int = 64,
        spectral_norm: bool = False,
        activation_function: str = "ReLU",
    ):
        super(ResidualStack, self).__init__()
        self.act_fn = getattr(nn, activation_function)()
        self.num_residual_blocks = num_residual_blocks
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    residual_hidden_channels=residual_hidden_channels,
                    spectral_norm=spectral_norm,
                    activation_function=activation_function,
                )
                for _ in range(self.num_residual_blocks)
            ]
        )

    def forward(self, x):
        for i in range(self.num_residual_blocks):
            x = self.layers[i](x)
        return self.act_fn(x)
