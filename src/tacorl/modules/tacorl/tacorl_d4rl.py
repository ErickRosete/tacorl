from pathlib import Path
from typing import List, Tuple

import hydra
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer

from tacorl.modules.cql.cql_offline_lightning_d4rl import CQL_Offline
from tacorl.utils.networks import (
    load_pl_module_from_checkpoint,
    set_parameter_requires_grad,
)


class TACORL(CQL_Offline):
    """Basic Actionable models implementation using PyTorch Lightning"""

    def __init__(
        self,
        play_lmp_dir: str = "~/tacorl/models/play_lmp",
        lmp_epoch_to_load: int = -1,
        overwrite_lmp_cfg: dict = {},
        finetune_action_decoder: bool = False,
        action_decoder_lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        # LMP Hyperparams
        self.play_lmp_dir = Path(play_lmp_dir).expanduser()
        self.lmp_epoch_to_load = lmp_epoch_to_load
        self.overwrite_lmp_cfg = overwrite_lmp_cfg

        # Taco-RL Hyperparams
        self.finetune_action_decoder = finetune_action_decoder
        self.action_decoder_lr = action_decoder_lr
        super().__init__(*args, **kwargs)

    def build_networks(self, *args, **kwargs):
        # Load LMP networks
        play_lmp = load_pl_module_from_checkpoint(
            self.play_lmp_dir,
            epoch=self.lmp_epoch_to_load,
            overwrite_cfg=self.overwrite_lmp_cfg,
        )
        self.action_decoder = play_lmp.action_decoder
        self.plan_recognition = play_lmp.plan_recognition

        # Build actor critic
        self.actor = play_lmp.plan_proposal
        critic_cfg = OmegaConf.to_container(self.critic_cfg, resolve=True)
        critic_cfg["q_network"]["num_layers"] = self.actor.policy.num_layers
        critic_cfg["q_network"]["hidden_dim"] = self.actor.policy.hidden_dim
        critic_cfg["state_dim"] = self.actor.state_dim
        critic_cfg["goal_dim"] = self.actor.goal_dim
        critic_cfg["action_dim"] = self.actor.action_dim
        self.q1 = hydra.utils.instantiate(critic_cfg)
        self.q2 = hydra.utils.instantiate(critic_cfg)
        self.target_q1 = hydra.utils.instantiate(critic_cfg)
        self.target_q2 = hydra.utils.instantiate(critic_cfg)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Make learned latent plans fixed
        set_parameter_requires_grad(self.plan_recognition, requires_grad=False)

    def get_rl_batch(self, batch, latent_plan):
        observations, actions, next_observations, rewards, dones = [], [], [], [], []
        for traj_idx in range(len(batch["observations"])):
            traj_observations = batch["observations"][traj_idx]
            goal = batch["goal"][traj_idx]
            obs = torch.cat([traj_observations[0], goal])
            next_obs = torch.cat([traj_observations[-1], goal])
            observations.append(obs)
            actions.append(latent_plan[traj_idx])
            next_observations.append(next_obs)
            success = int(batch["goal_reached"][traj_idx])
            rewards.append(success)
            dones.append(success)

        observations = torch.stack(observations, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        next_observations = torch.stack(next_observations, dim=0).float()
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(-1).float()
        dones = torch.tensor(dones, device=self.device).unsqueeze(-1)

        rl_batch = observations, actions, next_observations, rewards, dones
        return rl_batch

    def compute_action_decoder_loss(
        self, emb_states, actions, latent_plan, log_type="train"
    ) -> torch.Tensor:
        action_loss = self.action_decoder.loss(
            latent_plan=latent_plan,
            perceptual_emb=emb_states,
            actions=actions,
        )

        self.log(
            f"{log_type}/action_loss",
            action_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return action_loss

    def compute_action_decoder_update(
        self,
        states,
        actions,
        latent_plan,
        optimize: bool = True,
        log_type: str = "train",
    ):
        action_decoder_optimizer = self.optimizers()[-1]

        # From latent plan we can only infer actions to go from
        # S_t to S_g but not the action in S_g
        action_loss = self.compute_action_decoder_loss(
            emb_states=states[:, :-1],
            actions=actions[:, :-1],
            latent_plan=latent_plan,
            log_type=log_type,
        )

        if optimize:
            action_decoder_optimizer.zero_grad()
            self.manual_backward(action_loss)
            action_decoder_optimizer.step()

    @torch.no_grad()
    def get_pr_latent_plan(self, batch):
        self.plan_recognition.eval()
        pr_dist = self.plan_recognition(batch["observations"])
        latent_plan = pr_dist.sample()
        return latent_plan

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        # Get fixed latent plan
        latent_plan = self.get_pr_latent_plan(batch)
        self.compute_action_decoder_update(
            states=batch["observations"],
            actions=batch["actions"],
            latent_plan=latent_plan,
            optimize=self.finetune_action_decoder,
            log_type="train",
        )
        # Reinforcement learning
        rl_batch = self.get_rl_batch(batch, latent_plan)
        self.compute_update(rl_batch, optimize=True, log_type="train")

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        optimizers = super().configure_optimizers()
        if self.finetune_action_decoder:
            action_decoder_optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.action_decoder.parameters()),
                lr=self.action_decoder_lr,
            )
            optimizers.append(action_decoder_optimizer)
            return optimizers
        else:
            return optimizers
