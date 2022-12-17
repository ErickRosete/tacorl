from typing import Dict, List

import hydra
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer

from tacorl.utils.gym_utils import make_env


class RelayImitationLearning(pl.LightningModule):
    """Basic Relay Imitation Learning implementation using PyTorch Lightning"""

    def __init__(
        self,
        env,
        goal_encoder: DictConfig = {},
        perceptual_encoder: DictConfig = {},
        high_level_policy: DictConfig = {},
        low_level_policy: DictConfig = {},
        high_level_policy_modalities: List[str] = [],
        low_level_policy_modalities: List[str] = [],
        lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_cfg = env
        self.env = make_env(self.env_cfg)
        self.goal_encoder_cfg = goal_encoder
        self.perceptual_encoder_cfg = perceptual_encoder
        self.high_level_policy_cfg = high_level_policy
        self.high_level_policy_modalities = high_level_policy_modalities
        self.low_level_policy_cfg = low_level_policy
        self.low_level_policy_modalities = low_level_policy_modalities
        self.lr = lr
        all_modalities = high_level_policy_modalities + low_level_policy_modalities
        self.all_modalities = set(all_modalities)
        self.build_networks()
        self.save_hyperparameters()

    def build_networks(self):

        # Initialize perceptual_encoder for all modalities
        perceptual_encoder_cfg = OmegaConf.to_container(
            self.perceptual_encoder_cfg, resolve=True
        )
        for _, network_cfg in perceptual_encoder_cfg["networks"].items():
            if "device" in list(network_cfg.keys()):
                network_cfg["device"] = self.device

        perceptual_encoder_cfg["modalities"] = self.all_modalities
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder_cfg)

        # Goal encoder
        goal_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.all_modalities,
        )
        goal_encoder_cfg = OmegaConf.to_container(self.goal_encoder_cfg, resolve=True)
        goal_encoder_cfg["in_features"] = goal_dim
        self.goal_encoder = hydra.utils.instantiate(goal_encoder_cfg)

        # High level policy
        high_level_policy_state_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.high_level_policy_modalities,
        )
        update_dict = {
            "state_dim": high_level_policy_state_dim,
            "goal_dim": goal_encoder_cfg["out_features"],
        }
        high_level_policy_cfg = OmegaConf.to_container(
            self.high_level_policy_cfg, resolve=True
        )
        high_level_policy_cfg.update(update_dict)
        self.high_level_policy = hydra.utils.instantiate(high_level_policy_cfg)

        # Low level policy
        low_level_policy_state_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.low_level_policy_modalities,
        )
        update_dict = {
            "state_dim": low_level_policy_state_dim,
            "goal_dim": goal_encoder_cfg["out_features"],
        }
        low_level_policy_cfg = OmegaConf.to_container(
            self.low_level_policy_cfg, resolve=True
        )
        low_level_policy_cfg.update(update_dict)
        self.low_level_policy = hydra.utils.instantiate(low_level_policy_cfg)

    def get_emb_states(self, states, modalities: List[str] = []):
        emb_states = self.perceptual_encoder.get_state_from_observation(
            observation=states,
            modalities=modalities,
            cat_output=False,
        )
        return emb_states

    def compute_loss(self, batch, stage: str = "train") -> torch.Tensor:
        """Compute RIL Loss"""

        emb_states = self.get_emb_states(batch["obs"], modalities=self.all_modalities)
        low_level_emb_states = torch.cat(
            [emb_states[key] for key in self.low_level_policy_modalities],
            dim=-1,
        )
        high_level_emb_states = torch.cat(
            [emb_states[key] for key in self.high_level_policy_modalities],
            dim=-1,
        )

        # ======== Low level loss ===========
        # Low level goal
        emb_low_level_goal = self.get_emb_states(
            batch["low_level_goal"], modalities=self.all_modalities
        )
        emb_low_level_goal = torch.cat(
            [emb_low_level_goal[key] for key in self.low_level_policy_modalities],
            dim=-1,
        )
        emb_low_level_goal = self.goal_encoder(emb_low_level_goal)

        # Low level loss
        emb_representation = torch.cat(
            [low_level_emb_states, emb_low_level_goal], dim=-1
        )
        low_level_loss = -self.low_level_policy.log_prob(
            emb_representation, batch["low_level_action"]
        ).mean()

        self.log(
            f"{stage}/low_level_loss",
            low_level_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        # ========  High level ==============
        # High level goal
        emb_high_level_goal = self.get_emb_states(
            batch["high_level_goal"], modalities=self.all_modalities
        )
        emb_high_level_goal = torch.cat(
            [emb_high_level_goal[key] for key in self.high_level_policy_modalities],
            dim=-1,
        )
        emb_high_level_goal = self.goal_encoder(emb_high_level_goal)

        # High level action
        with torch.no_grad():
            emb_high_level_action = self.get_emb_states(
                batch["high_level_action"], modalities=self.all_modalities
            )
            emb_high_level_action = torch.cat(
                [
                    emb_high_level_action[key]
                    for key in self.high_level_policy_modalities
                ],
                dim=-1,
            )
            emb_high_level_action = self.goal_encoder(emb_high_level_action).detach()

        # High level loss
        emb_representation = torch.cat(
            [high_level_emb_states, emb_high_level_goal], dim=-1
        )
        high_level_loss = -self.high_level_policy.log_prob(
            emb_representation, emb_high_level_action
        ).mean()
        self.log(
            f"{stage}/high_level_loss",
            high_level_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        total_loss = low_level_loss + high_level_loss
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        """ """
        total_loss = self.compute_loss(batch, stage="train")
        self.log(
            "train/total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss

    def validation_step(  # type: ignore
        self,
        batch: Dict[
            str,
            Dict,
        ],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        batch:  Dict[str]( states: Dict[Tensor],
                           actions: Tensor,
                           info: Dict,
                           idx: int )
        """
        total_loss = self.compute_loss(batch, stage="validation")
        self.log(
            "validation/total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return total_loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        return optimizer
