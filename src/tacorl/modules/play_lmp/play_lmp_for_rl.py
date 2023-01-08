from typing import Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Independent, Normal
from torch.optim.optimizer import Optimizer

from tacorl.utils.distributions import TanhNormal
from tacorl.utils.episode_utils import get_state_info_on_idx, get_task_info_of_sequence
from tacorl.utils.gym_utils import make_env


class PlayLMP(pl.LightningModule):
    """LMP implementation using PyTorch Lightning"""

    def __init__(
        self,
        env: DictConfig = {},
        actor: DictConfig = {},
        plan_proposal: DictConfig = {},
        plan_recognition: DictConfig = {},
        perceptual_encoder: DictConfig = {},
        goal_encoder: DictConfig = {},
        action_decoder: DictConfig = {},
        transform_manager: DictConfig = {},
        dataloader: DictConfig = {},
        kl_beta: float = 1e-3,
        kl_balancing: bool = True,
        add_random_plan_loss: bool = False,
        kl_alpha: float = 0.8,
        lr: float = 1e-4,
        plan_proposal_obs_modalities: List[str] = [],
        plan_proposal_goal_modalities: List[str] = [],
        plan_recognition_modalities: List[str] = [],
        action_decoder_modalities: List[str] = [],
        real_world: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.real_world = real_world
        if not real_world:
            self.env_cfg = env
            self.env = make_env(self.env_cfg)

        self.add_random_plan_loss = add_random_plan_loss
        self.plan_proposal_obs_modalities = plan_proposal_obs_modalities
        self.plan_proposal_goal_modalities = plan_proposal_goal_modalities
        self.plan_recognition_modalities = plan_recognition_modalities
        self.action_decoder_modalities = action_decoder_modalities
        all_modalities = (
            plan_proposal_obs_modalities
            + plan_proposal_goal_modalities
            + plan_recognition_modalities
            + action_decoder_modalities
        )
        self.all_modalities = set(all_modalities)
        self.transform_manager = hydra.utils.instantiate(transform_manager)
        self.actor_cfg = actor
        self.plan_proposal_cfg = plan_proposal
        self.plan_recognition_cfg = plan_recognition
        self.perceptual_encoder_cfg = perceptual_encoder
        self.goal_encoder_cfg = goal_encoder
        self.action_decoder_cfg = action_decoder
        self.build_networks()
        self.lr = lr
        self.dataloader = dataloader
        self.kl_beta = kl_beta
        self.completed_tasks_by_idx = {}
        self.kl_balancing = kl_balancing
        self.kl_alpha = kl_alpha

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
        pp_state_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.plan_proposal_obs_modalities,
        )
        pp_goal_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.plan_proposal_goal_modalities,
        )
        # Goal encoder
        pr_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.plan_recognition_modalities,
        )
        goal_encoder_cfg = OmegaConf.to_container(self.goal_encoder_cfg, resolve=True)
        goal_encoder_cfg["in_features"] = pp_goal_dim
        goal_encoder_cfg["out_features"] = pp_goal_dim
        self.goal_encoder = hydra.utils.instantiate(goal_encoder_cfg)

        # Plan recognition
        plan_recognition_cfg = OmegaConf.to_container(
            self.plan_recognition_cfg, resolve=True
        )
        plan_recognition_cfg["state_dim"] = pr_dim
        self.plan_recognition = hydra.utils.instantiate(plan_recognition_cfg)

        # Plan proposal
        plan_proposal_cfg = OmegaConf.to_container(self.plan_proposal_cfg, resolve=True)
        plan_proposal_cfg["state_dim"] = pp_state_dim
        plan_proposal_cfg["goal_dim"] = goal_encoder_cfg["out_features"]
        if "Actor" in plan_proposal_cfg["_target_"].split(".")[-1]:
            plan_proposal_cfg["action_dim"] = self.plan_recognition.latent_plan_dim
        self.plan_proposal = hydra.utils.instantiate(plan_proposal_cfg)

        # Action decoder
        action_decoder_cfg = OmegaConf.to_container(
            self.action_decoder_cfg, resolve=True
        )
        ad_dim = self.perceptual_encoder.calc_state_dim(
            modalities=self.action_decoder_modalities,
        )
        action_decoder_cfg["state_dim"] = ad_dim
        action_decoder_cfg["goal_dim"] = goal_encoder_cfg["out_features"]
        self.action_decoder = hydra.utils.instantiate(action_decoder_cfg)

    def compute_action_loss(
        self,
        emb_states,
        actions,
        latent_plan,
        stage: str = "train",
        log_name_prefix: str = "",
        latent_goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_decoder.include_goal:
            action_loss, pred_actions = self.action_decoder.loss_and_act(
                latent_plan=latent_plan,
                perceptual_emb=emb_states,
                actions=actions,
                latent_goal=latent_goal,
            )
        else:
            # From latent plan we can only infer actions to go from
            # S_t to S_g but not the action in S_g
            action_loss, pred_actions = self.action_decoder.loss_and_act(
                latent_plan=latent_plan,
                perceptual_emb=emb_states[:, :-1],
                actions=actions[:, :-1],
            )

        self.log(
            f"{stage}/{log_name_prefix}action_loss",
            action_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Log gripper action accuracy
        pred_gripper_action = pred_actions[..., -1]
        pred_gripper_action = torch.where(pred_gripper_action > 0, 1.0, -1.0)
        pred_gripper_action[pred_gripper_action > 0] = 1
        gt_gripper_action = (
            actions[:, :-1, -1]
            if not self.action_decoder.include_goal
            else actions[..., -1]
        )
        gripper_accuracy = torch.mean(
            (gt_gripper_action == pred_gripper_action).float()
        )

        self.log(
            f"{stage}/{log_name_prefix}gripper_accuracy",
            gripper_accuracy,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return action_loss

    def get_emb_states(self, states, modalities: List[str] = []):
        bs, seq_len = list(states.values())[0].shape[:2]
        for key, value in states.items():
            states[key] = value.view(bs * seq_len, *value.shape[2:])
        emb_states = self.perceptual_encoder.get_state_from_observation(
            observation=states,
            modalities=modalities,
            cat_output=False,
        )
        for key, value in emb_states.items():
            emb_states[key] = value.view(bs, seq_len, -1)
        return emb_states

    def process_batch(self, batch):
        # Obtain plan proposal and plan recognition distributions
        emb_states = self.get_emb_states(
            batch["states"], modalities=self.all_modalities
        )
        pp_state = torch.cat(
            [emb_states[key][:, 0] for key in self.plan_proposal_obs_modalities], dim=-1
        )
        pp_goal = torch.cat(
            [emb_states[key][:, -1] for key in self.plan_proposal_goal_modalities],
            dim=-1,
        )
        pp_goal = self.goal_encoder(pp_goal)
        pp_dist = self.plan_proposal.get_dist(pp_state, pp_goal)
        pr_states = torch.cat(
            [emb_states[key] for key in self.plan_recognition_modalities],
            dim=-1,
        )
        pr_dist = self.plan_recognition(pr_states)
        return emb_states, pp_dist, pr_dist, pp_goal

    def compute_loss(self, batch, stage: str = "train") -> torch.Tensor:
        """Optimize latent plan with kl divergence"""
        emb_states, pp_dist, pr_dist, lat_goal = self.process_batch(batch)
        kl_loss = self.compute_kl_loss(
            pr_dist=pr_dist,
            pp_dist=pp_dist,
            stage=stage,
        )

        ad_states = torch.cat(
            [emb_states[key] for key in self.action_decoder_modalities],
            dim=-1,
        )

        action_loss = self.compute_action_loss(
            emb_states=ad_states,
            actions=batch["actions"],
            latent_plan=pr_dist.rsample(),
            stage=stage,
            latent_goal=lat_goal,
        )

        # Log random plan action loss
        random_plan = torch.empty_like(pr_dist.mean).uniform_(-1.0, 1.0)
        random_plan_action_loss = self.compute_action_loss(
            emb_states=ad_states,
            actions=batch["actions"],
            latent_plan=random_plan,
            stage=stage,
            log_name_prefix="random_plan_",
            latent_goal=torch.empty_like(lat_goal).uniform_(-1.0, 1.0),
        )

        total_loss = kl_loss + action_loss
        if self.add_random_plan_loss:
            total_loss -= random_plan_action_loss
        return total_loss, pp_dist

    def compute_kl_loss(
        self,
        pr_dist: torch.distributions.Distribution,
        pp_dist: torch.distributions.Distribution,
        stage: str = "train",
    ) -> torch.Tensor:

        if isinstance(pr_dist, TanhNormal):
            prior = pp_dist.normal
            posterior = pr_dist.normal
        else:
            prior = pp_dist
            posterior = pr_dist

        if self.kl_balancing:
            posterior_no_grad = Independent(
                Normal(loc=posterior.mean.detach(), scale=posterior.stddev.detach()), 1
            )
            prior_no_grad = Independent(
                Normal(loc=prior.mean.detach(), scale=prior.stddev.detach()), 1
            )
            kl_loss = (
                self.kl_alpha * D.kl_divergence(posterior_no_grad, prior).mean()
                + (1 - self.kl_alpha) * D.kl_divergence(posterior, prior_no_grad).mean()
            )
        else:
            kl_loss = D.kl_divergence(posterior, prior).mean()
        kl_loss_scaled = kl_loss * self.kl_beta
        self.log(
            f"{stage}/kl_loss",
            kl_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/kl_loss_scaled",
            kl_loss_scaled,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return kl_loss_scaled

    def set_kl_beta(self, kl_beta):
        """Set kl_beta from Callback"""
        self.kl_beta = kl_beta

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        """ """
        total_loss, pp_dist = self.compute_loss(batch, stage="train")
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
        total_loss, pp_dist = self.compute_loss(batch, stage="validation")
        self.log(
            "validation/total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        output = {
            "idx": batch["idx"],
            "sampled_plan_pp": pp_dist.sample(),
        }
        if not self.real_world:
            output["completed_tasks"] = self.get_completed_tasks(batch)
        return output

    def get_completed_tasks(self, batch):
        completed_tasks = []
        for i, idx in enumerate(batch["idx"]):
            if idx not in self.completed_tasks_by_idx:
                initial_state_info = get_state_info_on_idx(batch["state_info"], i, 0)
                last_state_info = get_state_info_on_idx(batch["state_info"], i, -1)
                self.completed_tasks_by_idx[idx] = get_task_info_of_sequence(
                    self.env, initial_state_info, last_state_info
                )
            completed_tasks.append(self.completed_tasks_by_idx[idx])
        return completed_tasks

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        return optimizer
