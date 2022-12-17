from typing import List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import cm
from omegaconf.dictconfig import DictConfig
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast


class TransformManager(object):
    def __init__(
        self,
        transforms: DictConfig = {},
    ) -> None:
        self.set_transforms(transforms)

    @torch.no_grad()
    def __call__(self, input: dict, transf_type="train", device="cpu"):
        """Transform state
        trans_type: 'train' or 'validation'
        """
        transf_input = {}
        for modality, value in input.items():
            value_tensor = self.to_tensor(value, modality=modality, device=device)
            if modality in self.transforms[transf_type].keys():
                transf_input[modality] = self.transforms[transf_type][modality](
                    value_tensor
                )
            else:
                transf_input[modality] = value_tensor
        return transf_input

    def to_tensor(self, value: np.ndarray, modality: str = "rgb", device: str = "cuda"):
        if torch.is_tensor(value):
            return value

        if "rgb" in modality or "depth" in modality:
            value_tensor = self.image_to_tensor(value, modality=modality, device=device)
        else:
            value_tensor = ArrayToTensor()(value, device=device)
        return value_tensor.float()

    def image_to_tensor(
        self, img: np.ndarray, modality: str = "rgb", device: str = "cuda"
    ):
        if "rgb" in modality:
            if len(img.shape) == 3:
                img = ArrayToTensor()(img, device=device)
                img = img.permute((2, 0, 1)).contiguous()
            elif len(img.shape) == 4:
                img = ArrayToTensor()(img, device=device)
                img = img.permute((0, 3, 1, 2)).contiguous()
        elif "depth" in modality:
            if len(img.shape) == 2 or len(img.shape) == 3:
                img = ArrayToTensor()(img, device=device)
            elif len(img.shape) == 4 and "depth_tactile" == modality:
                img = ArrayToTensor()(img, device=device)
                img = img.permute((0, 3, 1, 2)).contiguous()

        return img

    def set_transforms(self, transforms: dict):
        self.transforms = {}
        for transf_type, value in transforms.items():
            self.transforms[transf_type] = {}
            for modality, modality_transf in value.items():
                transforms_list = []
                for transf in modality_transf:
                    transforms_list.append(
                        hydra.utils.instantiate(transf, _convert_="partial")
                    )
                self.transforms[transf_type][modality] = self.get_transforms(
                    transforms_list
                )

    @staticmethod
    def get_transforms(transforms_list: List = None):
        if transforms_list is None:
            return ArrayToTensor()
        return transforms.Compose(transforms_list)


class ScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images to [0, 255] range
    Args:
        tensor (torch.tensor): Tensor to be scaled.
    Returns:
        Tensor: Scaled tensor.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)

        # Do not apply transform if image is already in float
        if 0.0 <= tensor.max() <= 1.0 and 0.0 <= tensor.min() <= 1.0:
            return tensor.float()
        return tensor.float().div(255).clip(0.0, 1.0)


class UpScaleImageTensor(object):
    """Scale tensor of shape (batch, C, H, W) containing images of  [0, 1] range to [0, 255]
    Args:
        tensor (torch.tensor): Tensor in range [0, 1].
    Returns:
        Tensor: Scaled tensor in [0, 255].
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        return tensor.float().mul(255)


class AdjustContrast(object):
    def __init__(self, contrast: float = 1.0):
        self.contrast = contrast

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return adjust_contrast(img, contrast_factor=0.5)

    def __repr__(self):
        return self.__class__.__name__ + +"(contrast={0})".format(self.contrast)


class ScaleDepthTensor(object):
    def __init__(self, min_depth: float = 0.01, max_depth: float = 2.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        normalized_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        return normalized_depth.clip(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + +"(min_depth={0}, max_depth={1})".format(
            self.min_depth, self.max_depth
        )


class LinearizeDepth(object):
    def __init__(self, near: float = 0.01, far: float = 2.0):
        self.far = far
        self.near = near

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        z_buffer = ((depth - self.near) * self.far) / ((self.far - self.near) * depth)
        lin_depth = (
            2.0 * self.near / (self.far + self.near - z_buffer * (self.far - self.near))
        )
        return lin_depth

    def __repr__(self):
        return self.__class__.__name__ + +"(near={0}, far={1})".format(
            self.near, self.far
        )


class ColorizeDepth(object):
    def __init__(self, colormap: str = " turbo"):
        self.colormap = cm.get_cmap(colormap)

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        if isinstance(depth, torch.Tensor):
            device = depth.device
            depth = depth.cpu().detach().numpy()
        assert np.issubdtype(
            depth.dtype, np.floating
        ), "Depth image is assumed to be float"
        assert len(depth.shape) == 3, "Depth image doesn't have the appropiate shape"
        depth = depth.clip(0, 1)
        colorized_depth = self.colormap(depth)[:, :, :, :3]
        colorized_depth = (
            torch.from_numpy(colorized_depth).to(device).permute(0, 3, 1, 2)
        )
        return colorized_depth.squeeze()

    def __repr__(self):
        return self.__class__.__name__ + "(colormap={0})".format(self.colormap)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.Tensor(std)
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        device = tensor.device
        if device != self.std.device:
            self.std = self.std.to(device)
        if device != self.mean.device:
            self.mean = self.mean.to(device)
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class AddDepthNoise(object):
    """Add multiplicative gamma noise to depth image.
    This is adapted from the DexNet 2.0 code.
    Their code:
    https://github.com/BerkeleyAutomation/gqcnn/blob/master/gqcnn/training/tf/trainer_tf.py
    """

    def __init__(self, shape=1000.0, rate=1000.0):
        self.shape = torch.tensor(shape)
        self.rate = torch.tensor(rate)
        self.dist = torch.distributions.gamma.Gamma(
            torch.tensor(shape), torch.tensor(rate)
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        multiplicative_noise = self.dist.sample()
        return multiplicative_noise * tensor

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"shape={self.shape}, rate={self.rate}, dist={self.dist}"
        )


class ArrayToTensor(object):
    """Transforms np array to tensor."""

    def __call__(
        self, array: np.ndarray, device: torch.device = "cuda"
    ) -> torch.Tensor:
        assert isinstance(array, np.ndarray)
        return torch.from_numpy(array).to(device)


class NormalizeVector(object):
    """Normalize a tensor vector with mean and standard deviation."""

    def __init__(self, mean=0.0, std=1.0):
        if isinstance(mean, float):
            mean = [mean]
        if isinstance(std, float):
            std = [std]
        self.std = torch.Tensor(std)
        self.std[self.std == 0.0] = 1.0
        self.mean = torch.Tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        device = tensor.device
        self.std = self.std.to(device)
        self.mean = self.mean.to(device)
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        input_dim = x.dim()
        if input_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif input_dim == 3:
            x = x.unsqueeze(0)
        n, c, h, w = x.size()
        assert h == w, "Height and width"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        if input_dim == 2:
            output = output.squeeze(0).squeeze(0)
        elif input_dim == 3:
            output = output.squeeze(0)
        return output


class ColorTransform(object):
    def __init__(self, contrast=0.3, brightness=0.3, hue=0.3, prob=1.0):
        super().__init__()
        self.prob = prob
        self.jitter = transforms.ColorJitter(
            contrast=contrast, brightness=brightness, hue=hue
        )

    def apply_transform(self, single_img):
        if np.random.rand() < self.prob:
            tensor = self.jitter(single_img)
        return tensor

    def apply_transform_batch(self, imgs):
        """Apply different transform to each image in batch"""
        transformed_images = []
        for img in imgs:
            transformed_images.append(self.apply_transform(img))
        return torch.stack(transformed_images)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            tensor, torch.Tensor
        ), "Input of color transform must be a tensor"
        if len(tensor.shape) == 4:
            return self.apply_transform_batch(tensor)
        elif len(tensor.shape) == 3:
            return self.apply_transform(tensor)
        return tensor


def transform_observation(
    transform_manager: TransformManager,
    obs: dict,
    transf_type: str = "validation",
    device: str = "cuda",
):
    if isinstance(obs, dict) and "goal" in obs:
        transf_obs = transform_manager(
            obs["observation"], transf_type=transf_type, device=device
        )
        transf_goal = transform_manager(
            obs["goal"], transf_type=transf_type, device=device
        )
        return {"observation": transf_obs, "goal": transf_goal}
    else:
        return transform_manager(obs, transf_type=transf_type, device=device)
