from typing import Dict, List

import d4rl  # noqa
import gym
import hydra
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Independent, Normal
from torch.optim.optimizer import Optimizer

from tacorl.utils.distributions import TanhNormal


class PlayLMP(pl.LightningModule):
    """LMP implementation using PyTorch Lightning"""

    def __init__(
        self,
        actor: DictConfig = {},
        plan_proposal: DictConfig = {},
        plan_recognition: DictConfig = {},
        action_decoder: DictConfig = {},
        transform_manager: DictConfig = {},
        dataloader: DictConfig = {},
        kl_beta: float = 1e-3,
        kl_balancing: bool = True,
        add_random_plan_loss: bool = False,
        kl_alpha: float = 0.8,
        lr: float = 1e-4,
        d4rl_env: str = "antmaze-large-diverse-v0",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.add_random_plan_loss = add_random_plan_loss
        self.transform_manager = hydra.utils.instantiate(transform_manager)
        self.actor_cfg = actor
        self.plan_proposal_cfg = plan_proposal
        self.plan_recognition_cfg = plan_recognition
        self.action_decoder_cfg = action_decoder
        self.env = gym.make(d4rl_env)
        self.build_networks()
        self.lr = lr
        self.dataloader = dataloader
        self.kl_beta = kl_beta
        self.completed_tasks_by_idx = {}
        self.kl_balancing = kl_balancing
        self.kl_alpha = kl_alpha

        self.save_hyperparameters()

    def build_networks(self):
        goal_dim = 2
        state_dim = self.env.observation_space.shape[0]
        self.action_decoder_cfg["out_features"] = self.env.action_space.shape[0]
        self.action_decoder_cfg["act_max_bound"] = self.env.action_space.high.tolist()
        self.action_decoder_cfg["act_min_bound"] = self.env.action_space.low.tolist()

        # Plan recognition
        plan_recognition_cfg = OmegaConf.to_container(
            self.plan_recognition_cfg, resolve=True
        )
        plan_recognition_cfg["state_dim"] = state_dim
        self.plan_recognition = hydra.utils.instantiate(plan_recognition_cfg)

        # Plan proposal
        plan_proposal_cfg = OmegaConf.to_container(self.plan_proposal_cfg, resolve=True)
        plan_proposal_cfg["state_dim"] = state_dim
        plan_proposal_cfg["goal_dim"] = goal_dim
        if "Actor" in plan_proposal_cfg["_target_"].split(".")[-1]:
            plan_proposal_cfg["action_dim"] = self.plan_recognition.latent_plan_dim
        self.plan_proposal = hydra.utils.instantiate(plan_proposal_cfg)

        # Action decoder
        action_decoder_cfg = OmegaConf.to_container(
            self.action_decoder_cfg, resolve=True
        )
        action_decoder_cfg["state_dim"] = state_dim
        self.action_decoder = hydra.utils.instantiate(action_decoder_cfg)

    def compute_action_loss(
        self,
        emb_states,
        actions,
        latent_plan,
        stage: str = "train",
        log_name: str = "action_loss",
    ) -> torch.Tensor:

        action_loss = self.action_decoder.loss(
            latent_plan=latent_plan,
            perceptual_emb=emb_states,
            actions=actions,
        )

        self.log(
            f"{stage}/{log_name}",
            action_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return action_loss

    def process_batch(self, batch):
        # Obtain plan proposal and plan recognition distributions
        pp_state = batch["observations"][:, 0]
        pp_goal = batch["observations"][:, -1, :2]
        pp_dist = self.plan_proposal.get_dist(pp_state, pp_goal)
        pr_states = batch["observations"]
        pr_dist = self.plan_recognition(pr_states)
        return pp_dist, pr_dist

    def compute_loss(self, batch, stage: str = "train") -> torch.Tensor:
        """Optimize latent plan with kl divergence"""
        pp_dist, pr_dist = self.process_batch(batch)
        kl_loss = self.compute_kl_loss(
            pr_dist=pr_dist,
            pp_dist=pp_dist,
            stage=stage,
        )
        # From latent plan we can only infer actions to go from
        # S_t to S_g but not the action in S_g
        action_loss = self.compute_action_loss(
            emb_states=batch["observations"][:, :-1],
            actions=batch["actions"][:, :-1],
            latent_plan=pr_dist.rsample(),
            stage=stage,
        )
        random_plan = torch.empty_like(pr_dist.mean).uniform_(-1.0, 1.0)
        random_plan_action_loss = self.compute_action_loss(
            emb_states=batch["observations"][:, :-1],
            actions=batch["actions"][:, :-1],
            latent_plan=random_plan,
            stage=stage,
            log_name="random_plan_action_loss",
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
        return output

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        return optimizer
