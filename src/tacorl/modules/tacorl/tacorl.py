import copy
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from torch.optim.optimizer import Optimizer

from tacorl.modules.cql.cql_offline_lightning import CQL_Offline
from tacorl.networks.actor_critic.visual_actor_wrapper import VisualActorWrapper
from tacorl.networks.actor_critic.visual_critic_wrapper import VisualCriticWrapper
from tacorl.utils.misc import list_of_dicts_to_dict_tensor, transform_to_list
from tacorl.utils.networks import (
    load_pl_module_from_checkpoint,
    set_parameter_requires_grad,
)


class TACORL(CQL_Offline):
    """Basic Actionable models implementation using PyTorch Lightning"""

    def __init__(
        self,
        play_lmp_dir: str = "~/thesis/models/play_lmp",
        finetune_action_decoder: bool = False,
        action_decoder_lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        self.finetune_action_decoder = finetune_action_decoder
        self.action_decoder_lr = action_decoder_lr
        self.play_lmp_dir = Path(play_lmp_dir).expanduser()
        super().__init__(*args, **kwargs)

    def build_networks(self):
        # Load LMP networks
        play_lmp = load_pl_module_from_checkpoint(self.play_lmp_dir)
        self.action_decoder = play_lmp.action_decoder
        self.perceptual_encoder = play_lmp.perceptual_encoder
        self.plan_recognition = play_lmp.plan_recognition
        self.action_decoder_modalities = play_lmp.action_decoder_modalities
        self.plan_recognition_modalities = play_lmp.plan_recognition_modalities
        self.all_modalities = list(
            set(self.action_decoder_modalities + self.plan_recognition_modalities)
        )
        env_modalities = play_lmp.plan_proposal_obs_modalities
        goal_modalities = play_lmp.plan_proposal_goal_modalities

        # Build actor critic
        actor = play_lmp.plan_proposal
        self.actor = VisualActorWrapper(
            encoder=copy.deepcopy(self.perceptual_encoder),
            goal_encoder=copy.deepcopy(play_lmp.goal_encoder),
            actor=actor,
            env_modalities=env_modalities,
            goal_modalities=goal_modalities,
        )
        critic_cfg = OmegaConf.to_container(self.critic_cfg, resolve=True)
        critic_cfg["q_network"]["num_layers"] = actor.policy.num_layers
        critic_cfg["q_network"]["hidden_dim"] = actor.policy.hidden_dim
        critic_cfg["state_dim"] = actor.state_dim
        critic_cfg["goal_dim"] = actor.goal_dim
        critic_cfg["action_dim"] = actor.action_dim

        critic_encoder_cfg = OmegaConf.to_container(
            self.critic_encoder_cfg, resolve=True
        )
        critic_encoder_cfg["modalities"] = list(set(env_modalities + goal_modalities))
        for modality, network_cfg in critic_encoder_cfg["networks"].items():
            if "device" in list(network_cfg.keys()):
                network_cfg["device"] = self.device
            # Make critic encoder latent dim consistent with play lmp goal encoder
            if (
                "latent_dim" in list(network_cfg.keys())
                and modality in self.perceptual_encoder.networks
            ):
                network_cfg["latent_dim"] = self.perceptual_encoder.networks[
                    modality
                ].latent_dim
        self.q1 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=copy.deepcopy(play_lmp.goal_encoder),
            env_modalities=env_modalities,
            goal_modalities=goal_modalities,
        )
        self.q2 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=copy.deepcopy(play_lmp.goal_encoder),
            env_modalities=env_modalities,
            goal_modalities=goal_modalities,
        )
        self.target_q1 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=copy.deepcopy(play_lmp.goal_encoder),
            env_modalities=env_modalities,
            goal_modalities=goal_modalities,
        )
        self.target_q2 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=copy.deepcopy(play_lmp.goal_encoder),
            env_modalities=env_modalities,
            goal_modalities=goal_modalities,
        )
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # Make learned latent plans fixed
        set_parameter_requires_grad(self.perceptual_encoder, requires_grad=False)
        set_parameter_requires_grad(self.plan_recognition, requires_grad=False)

    def get_emb_states(self, states, modalities: List[str] = []):
        rs_states = {}
        bs, seq_len = list(states.values())[0].shape[:2]
        for key, value in states.items():
            rs_states[key] = value.view(bs * seq_len, *value.shape[2:])
        emb_states = self.perceptual_encoder.get_state_from_observation(
            observation=rs_states,
            modalities=modalities,
            cat_output=False,
        )
        for key, value in emb_states.items():
            emb_states[key] = value.view(bs, seq_len, -1)
        return emb_states

    def get_rl_batch(self, batch, latent_plan):
        actions, rewards, dones = [], [], []
        states_observations, states_goals = [], []
        next_states_observations, next_states_goals = [], []

        trajectories_states = transform_to_list(batch["states"])
        goals = transform_to_list(batch["goal"])
        for traj_idx in range(len(trajectories_states)):
            # Separate to a list, where each element is a state of the trajectory
            states_in_traj = transform_to_list(trajectories_states[traj_idx])
            goal = goals[traj_idx]
            # Goal reached
            states_observations.append(states_in_traj[0])
            states_goals.append(goal)
            next_states_observations.append(states_in_traj[-1])
            next_states_goals.append(goal)
            actions.append(latent_plan[traj_idx])
            success = int(batch["disp"][traj_idx] == 1)
            rewards.append(success)
            dones.append(success)

        states_observations = list_of_dicts_to_dict_tensor(states_observations)
        states_goals = list_of_dicts_to_dict_tensor(states_goals)
        next_states_observations = list_of_dicts_to_dict_tensor(
            next_states_observations
        )
        next_states_goals = list_of_dicts_to_dict_tensor(next_states_goals)
        states = {"observation": states_observations, "goal": states_goals}
        next_states = {
            "observation": next_states_observations,
            "goal": next_states_goals,
        }
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(-1)
        dones = torch.tensor(dones, device=self.device).unsqueeze(-1)

        rl_batch = states, actions, next_states, rewards, dones
        return rl_batch

    def expand_emb(self, emb, seq_len):
        emb = emb.expand(seq_len, *emb.shape)  # seq_len, bs, emb_size
        emb = emb.permute(1, 0, 2)  # bs, seq_len, emb_size
        emb = emb.reshape(-1, emb.shape[-1])  # bs * seq_len, emb_size
        return emb

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
        emb_states,
        actions,
        latent_plan,
        optimize: bool = True,
        log_type: str = "train",
    ):
        action_decoder_optimizer = self.optimizers()[-1]

        # Imitation learning
        ad_states = torch.cat(
            [emb_states[key] for key in self.action_decoder_modalities],
            dim=-1,
        )
        # From latent plan we can only infer actions to go from
        # S_t to S_g but not the action in S_g
        action_loss = self.compute_action_decoder_loss(
            emb_states=ad_states[:, :-1],
            actions=actions[:, :-1],
            latent_plan=latent_plan,
            log_type=log_type,
        )

        if optimize:
            action_decoder_optimizer.zero_grad()
            self.manual_backward(action_loss)
            action_decoder_optimizer.step()

    def get_pr_latent_plan(self, batch, return_emb_states=True):
        with torch.no_grad():
            self.perceptual_encoder.eval()
            self.plan_recognition.eval()
            emb_states = self.get_emb_states(
                batch["states"], modalities=self.all_modalities
            )
            pr_states = torch.cat(
                [emb_states[key] for key in self.plan_recognition_modalities],
                dim=-1,
            )
            pr_dist = self.plan_recognition(pr_states)
            latent_plan = pr_dist.sample()

        if return_emb_states:
            return latent_plan, emb_states
        else:
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
        latent_plan, emb_states = self.get_pr_latent_plan(batch, return_emb_states=True)
        self.compute_action_decoder_update(
            emb_states=emb_states,
            actions=batch["actions"],
            latent_plan=latent_plan,
            optimize=self.finetune_action_decoder,
            log_type="train",
        )
        # Reinforcement learning
        rl_batch = self.get_rl_batch(batch, latent_plan)
        self.compute_update(rl_batch, optimize=True, log_type="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs
    ):
        latent_plan, emb_states = self.get_pr_latent_plan(batch, return_emb_states=True)
        self.compute_action_decoder_update(
            emb_states=emb_states,
            actions=batch["actions"],
            latent_plan=latent_plan,
            optimize=False,
            log_type="validation",
        )
        rl_batch = self.get_rl_batch(batch, latent_plan)
        self.compute_update(rl_batch, optimize=False, log_type="validation")

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
