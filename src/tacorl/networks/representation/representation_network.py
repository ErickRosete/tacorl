from collections import OrderedDict
from typing import List

import hydra
import torch
import torch.nn as nn
from omegaconf.omegaconf import DictConfig


class ForwardModelNetwork(nn.Module):
    """Predicts the latent representation of the next obs
    given the current obs latent representation and the action
    """

    def __init__(
        self,
        visual_state_dim: int = 16,
        action_dim: int = 0,
        hidden_dim: int = 256,
        activation_function: str = "ReLU",
    ):
        super(ForwardModelNetwork, self).__init__()
        self.act_fn = getattr(nn, activation_function)()

        self.model = nn.Sequential(
            nn.Linear(visual_state_dim + action_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, visual_state_dim),
        )

    def forward(self, visual_state: torch.Tensor, action: torch.Tensor):
        input = torch.cat([visual_state, action], dim=-1)
        return self.model(input)


class InverseModelNetwork(nn.Module):
    """Predicts the latent representation of the next obs
    given the current obs latent representation and the action
    """

    def __init__(
        self,
        visual_state_dim: int = 16,
        action_dim: int = 0,
        hidden_dim: int = 256,
        activation_function: str = "ReLU",
    ):
        super(InverseModelNetwork, self).__init__()
        self.act_fn = getattr(nn, activation_function)()

        self.model = nn.Sequential(
            OrderedDict(
                nn.Linear(visual_state_dim * 2, hidden_dim),
                self.act_fn,
                nn.Linear(hidden_dim, action_dim),
            )
        )

    def forward(self, visual_state: torch.Tensor, next_visual_state: torch.Tensor):
        input = torch.cat([visual_state, next_visual_state], dim=-1)
        return self.model(input)


class ProprioReconstNetwork(nn.Module):
    """Reconstruct proprioceptive information
    from visual state representation
    """

    def __init__(
        self,
        visual_state_dim: int = 16,
        proprioceptive_dim: int = 15,
        hidden_dim: int = 256,
        activation_function: str = "ReLU",
    ):
        super(ProprioReconstNetwork, self).__init__()
        self.act_fn = getattr(nn, activation_function)()

        self.model = nn.Sequential(
            nn.Linear(visual_state_dim, hidden_dim),
            self.act_fn,
            nn.Linear(hidden_dim, proprioceptive_dim),
        )

    def forward(self, visual_state: torch.Tensor):
        return self.model(visual_state)


class LateFusion(nn.Module):
    """Representation learning network (Encoder)
    Contains dictionary with one network per image modality
    """

    def __init__(self, networks: DictConfig = {}, modalities: List[str] = []):
        super(LateFusion, self).__init__()
        for modality in modalities:
            assert (
                modality in networks.keys()
            ), f"Network configuration for {modality} is missing"

        networks_aux = {}
        self.visual_state_dim = 0
        for modality, encoder_cfg in networks.items():
            if modality in modalities:
                networks_aux[modality] = hydra.utils.instantiate(encoder_cfg)
                self.visual_state_dim += networks_aux[modality].latent_dim
        self.networks = nn.ModuleDict(networks_aux)

    def forward(self, inputs):
        outputs = {}
        for modality in inputs.keys():
            if modality in self.networks.keys():
                outputs[modality] = self.networks[modality](inputs[modality])
        return outputs

    def get_state_from_observation(
        self,
        observation: dict,
        modalities: List[str] = [],
        cat_output: bool = True,
    ):
        """Uses the vision networks to obtain a compact representation of
        the observation"""
        # Gym envs
        if not isinstance(observation, dict):
            return observation

        # Robot envs
        state = {}
        for modality in modalities:
            if "rgb" in modality or "depth" in modality:
                # Image modalities
                img = observation[modality]
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                state[modality] = self.networks[modality](img).squeeze(0)
            else:
                # Vector modalities
                if isinstance(observation[modality], List):
                    observation[modality] = torch.stack(observation[modality], dim=-1)
                state[modality] = observation[modality].float()

        if cat_output:
            state = torch.cat(list(state.values()), dim=-1)
        return state

    def calc_state_dim(self, modalities: List[str] = []):
        state_dim = 0
        for modality in modalities:
            state_dim += self.networks[modality].latent_dim
        return state_dim
