from typing import Dict, List

import hydra
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer

from tacorl.utils.episode_utils import get_state_info_on_idx, get_task_info_of_sequence
from tacorl.utils.gym_utils import get_env_info, make_env


class PlayLMP(pl.LightningModule):
    """Basic Actionable models implementation using PyTorch Lightning"""

    def __init__(
        self,
        env: DictConfig = {},
        actor: DictConfig = {},
        plan_proposal: DictConfig = {},
        plan_recognition: DictConfig = {},
        transform_manager: DictConfig = {},
        dataloader: DictConfig = {},
        kl_beta: float = 1e-3,
        lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_cfg = env
        self.env = make_env(self.env_cfg)
        self.transform_manager = hydra.utils.instantiate(transform_manager)
        self.actor_cfg = actor
        self.plan_proposal_cfg = plan_proposal
        self.plan_recognition_cfg = plan_recognition
        self.build_networks()
        self.lr = lr
        self.dataloader = dataloader
        self.kl_beta = kl_beta
        self.completed_tasks_by_idx = {}
        self.save_hyperparameters()

    def build_networks(self):
        env_info = get_env_info(self.env)
        self.actor_cfg = OmegaConf.to_container(self.actor_cfg, resolve=True)
        self.actor_cfg.update(env_info)
        self.actor = hydra.utils.instantiate(self.actor_cfg)
        plan_proposal_cfg = OmegaConf.to_container(self.plan_proposal_cfg, resolve=True)
        plan_proposal_cfg["state_dim"] = self.actor.state_dim
        plan_proposal_cfg["goal_dim"] = self.actor.goal_dim
        self.plan_proposal = hydra.utils.instantiate(plan_proposal_cfg)
        plan_recognition_cfg = OmegaConf.to_container(
            self.plan_recognition_cfg, resolve=True
        )
        plan_recognition_cfg["state_dim"] = self.actor.state_dim
        self.plan_recognition = hydra.utils.instantiate(plan_recognition_cfg)

    def compute_action_loss(
        self, emb_states, actions, pr_dist, stage: str = "train"
    ) -> torch.Tensor:
        latent_plan = pr_dist.rsample()
        latent_plan = latent_plan.expand(actions.shape[1], *latent_plan.shape)
        latent_plan = latent_plan.permute(1, 0, 2)
        latent_plan = latent_plan.reshape(-1, latent_plan.shape[-1])

        emb_goals = emb_states[:, -1]
        emb_goals = emb_goals.expand(actions.shape[1], *emb_goals.shape)
        emb_goals = emb_goals.permute(1, 0, 2)
        emb_goals = emb_goals.reshape(-1, emb_goals.shape[-1])

        emb_states = emb_states.reshape(-1, emb_states.shape[-1])
        emb_states = torch.cat((emb_states, emb_goals), dim=-1)

        # Min log likelihood of predicted actions
        _, log_pi = self.actor.get_actions(
            observation=emb_states, reparameterize=True, latent_plan=latent_plan
        )

        # Max log likelihood of dataset actions
        actions = actions.reshape(-1, actions.shape[-1])
        policy_log_prob = self.actor.log_prob(
            observations=emb_states, actions=actions, latent_plan=latent_plan
        )

        # Loss function
        action_loss = (log_pi - policy_log_prob).mean()

        self.log(
            f"{stage}/action_loss",
            action_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return action_loss

    def get_emb_states(self, states):
        bs, seq_len = list(states.values())[0].shape[:2]
        for key, value in states.items():
            states[key] = value.view(bs * seq_len, *value.shape[2:])
        emb_states = self.actor.get_emb_representation(states, cat_output=False)
        emb_states = emb_states.view(bs, seq_len, -1)
        return emb_states

    def process_batch(self, batch):
        # Obtain plan proposal and plan recognition distributions
        emb_states = self.get_emb_states(batch["states"])
        pp_dist = self.plan_proposal(emb_states[:, 0], emb_states[:, -1])
        pr_dist = self.plan_recognition(emb_states)
        return emb_states, pp_dist, pr_dist

    def compute_loss(self, batch, stage: str = "train") -> torch.Tensor:
        """Optimize latent plan with kl divergence"""
        emb_states, pp_dist, pr_dist = self.process_batch(batch)
        kl_loss = self.compute_kl_loss(
            pr_dist=pr_dist,
            pp_dist=pp_dist,
            stage=stage,
        )
        action_loss = self.compute_action_loss(
            emb_states=emb_states,
            actions=batch["actions"],
            pr_dist=pr_dist,
            stage=stage,
        )
        total_loss = kl_loss + action_loss
        return total_loss, pp_dist

    def compute_kl_loss(
        self,
        pr_dist: torch.distributions.Distribution,
        pp_dist: torch.distributions.Distribution,
        stage: str = "train",
    ) -> torch.Tensor:
        kl_loss = D.kl_divergence(pr_dist, pp_dist).mean()
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
        completed_tasks = self.get_completed_tasks(batch)
        # emb_states, pp_dist, pr_dist = self.process_batch(batch)
        output = {
            "idx": batch["idx"],
            "completed_tasks": completed_tasks,
            "sampled_plan_pp": pp_dist.sample(),
        }
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
