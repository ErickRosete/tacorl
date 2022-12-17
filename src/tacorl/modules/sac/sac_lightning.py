import logging
from collections import deque, namedtuple
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from tacorl.modules.sac.replay_buffer import ReplayBuffer
from tacorl.modules.sac.sac_agent import SAC_Agent
from tacorl.networks.actor_critic.visual_actor_wrapper import VisualActorWrapper
from tacorl.networks.actor_critic.visual_critic_wrapper import VisualCriticWrapper
from tacorl.utils.gym_utils import get_env_info, make_env, make_env_fn
from tacorl.utils.misc import dict_to_list_of_dicts

SACLosses = namedtuple(
    "SACLosses",
    "actor_loss critic_loss alpha_loss",
)


class SAC(pl.LightningModule):
    """Basic SAC implementation using PyTorch Lightning"""

    def __init__(
        self,
        env: DictConfig = {},
        actor: DictConfig = {},
        critic: DictConfig = {},
        actor_encoder: DictConfig = {},
        critic_encoder: DictConfig = {},
        goal_encoder: DictConfig = {},
        transform_manager: DictConfig = {},
        discount: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        deterministic_backup: bool = False,
        num_parallel_envs: int = 1,
        replay_buffer_size: int = 5e6,
        populate_replay_buffer: bool = True,
        fill_strategy: str = "random",
        replay_buffer_path: str = None,
        warm_start_steps: int = 1000,
        model_dir: Path = None,
        reward_scale: float = 1.0,
        bc_epochs: int = 0,
        clip_grad: bool = True,
        clip_grad_val: int = 1,
    ) -> None:

        super().__init__()
        # Instantiate train environment
        self.env_cfg = env
        self.env = make_env(self.env_cfg)
        self.transform_manager = hydra.utils.instantiate(transform_manager)
        # Log
        self.cons_logger = logging.getLogger(__name__)
        # Hyperparameters
        self.deterministic_backup = deterministic_backup
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_cfg = actor
        self.critic_cfg = critic
        self.actor_encoder_cfg = actor_encoder
        self.critic_encoder_cfg = critic_encoder
        self.goal_encoder_cfg = goal_encoder
        self.discount = discount
        self.reward_scale = reward_scale
        self.tau = tau
        self.model_dir = model_dir
        self.bc_epochs = bc_epochs
        self.clip_grad = clip_grad
        self.clip_grad_val = clip_grad_val
        # Environment
        self.replay_buffer_path = replay_buffer_path
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.agent = SAC_Agent(
            env=self.env,
            replay_buffer=self.replay_buffer,
            transform_manager=self.transform_manager,
        )
        # Entropy
        self.target_entropy = -np.prod(
            self.env.action_space.shape
        ).item()  # heuristic value
        self.log_alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.build_networks()
        # Load replay buffer
        successful_load = self.replay_buffer.load(replay_buffer_path)
        if populate_replay_buffer and not successful_load:
            self.populate(
                warm_start_steps, fill_strategy, num_parallel_envs=num_parallel_envs
            )
        # Logic values
        self.episode_number = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.episode_step = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.episode_return, self.episode_length = 0.0, 0.0
        self.episodes_returns, self.episodes_lengths = deque(maxlen=int(10)), deque(
            maxlen=int(10)
        )
        self.accuracies = deque(maxlen=int(10))
        self.rewards = deque(maxlen=int(1000))

        # PyTorch lightning
        self.automatic_optimization = False
        self.save_hyperparameters()

    def build_networks(self):
        env_info = get_env_info(self.env)

        all_modalities = env_info["env_modalities"] + env_info["goal_modalities"]
        all_modalities = set(all_modalities)
        actor_encoder_cfg = OmegaConf.to_container(self.actor_encoder_cfg, resolve=True)
        actor_encoder_cfg["modalities"] = all_modalities
        actor_encoder = hydra.utils.instantiate(actor_encoder_cfg)
        state_dim = actor_encoder.calc_state_dim(
            modalities=env_info["env_modalities"],
        )
        goal_dim = actor_encoder.calc_state_dim(
            modalities=env_info["goal_modalities"],
        )

        goal_encoder_cfg = OmegaConf.to_container(self.goal_encoder_cfg, resolve=True)
        goal_encoder_cfg["in_features"] = goal_dim
        goal_encoder_cfg["out_features"] = goal_dim

        update_dict = {
            "state_dim": state_dim,
            "goal_dim": goal_encoder_cfg["out_features"],
            "action_dim": env_info["action_dim"],
        }
        actor_cfg = OmegaConf.to_container(self.actor_cfg, resolve=True)
        actor_cfg.update(update_dict)
        critic_cfg = OmegaConf.to_container(self.critic_cfg, resolve=True)
        critic_cfg.update(update_dict)

        self.actor = VisualActorWrapper(
            actor=hydra.utils.instantiate(actor_cfg),
            encoder=actor_encoder,
            goal_encoder=hydra.utils.instantiate(goal_encoder_cfg),
            env_modalities=env_info["env_modalities"],
            goal_modalities=env_info["goal_modalities"],
        )

        critic_encoder_cfg = OmegaConf.to_container(
            self.critic_encoder_cfg, resolve=True
        )
        critic_encoder_cfg["modalities"] = all_modalities

        self.q1 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=hydra.utils.instantiate(goal_encoder_cfg),
            env_modalities=env_info["env_modalities"],
            goal_modalities=env_info["goal_modalities"],
        )
        self.q2 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=hydra.utils.instantiate(goal_encoder_cfg),
            env_modalities=env_info["env_modalities"],
            goal_modalities=env_info["goal_modalities"],
        )
        self.target_q1 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=hydra.utils.instantiate(goal_encoder_cfg),
            env_modalities=env_info["env_modalities"],
            goal_modalities=env_info["goal_modalities"],
        )
        self.target_q2 = VisualCriticWrapper(
            critic=hydra.utils.instantiate(critic_cfg),
            encoder=hydra.utils.instantiate(critic_encoder_cfg),
            goal_encoder=hydra.utils.instantiate(goal_encoder_cfg),
            env_modalities=env_info["env_modalities"],
            goal_modalities=env_info["goal_modalities"],
        )
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    @staticmethod
    def soft_update_from_to(source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def compute_update(self, batch, optimize: bool = True):
        alpha_optimizer, actor_optimizer, q1_optimizer, q2_optimizer = self.optimizers()
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
            batch, alpha_optimizer, optimize=optimize
        )

        q1_loss, q2_loss = self.compute_critic_loss(batch)

        # Logs
        self.log("q1_loss", q1_loss, on_step=True)
        self.log("q2_loss", q2_loss, on_step=True)
        self.log("actor_loss", actor_loss, on_step=True)
        self.log("alpha_loss", alpha_loss, on_step=True)

        # Optimize
        if optimize:
            q1_optimizer.zero_grad()
            self.manual_backward(q1_loss)
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.clip_grad_val)
            q1_optimizer.step()

            q2_optimizer.zero_grad()
            self.manual_backward(q2_loss)
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.clip_grad_val)
            q2_optimizer.step()

            actor_optimizer.zero_grad()
            self.manual_backward(actor_loss)
            actor_optimizer.step()

            # Soft updates
            self.soft_update_from_to(self.q1, self.target_q1, self.tau)
            self.soft_update_from_to(self.q2, self.target_q2, self.tau)

    def compute_critic_loss(self, batch):
        (
            batch_observations,
            batch_actions,
            batch_next_observations,
            batch_rewards,
            batch_dones,
        ) = batch

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_actions(
                batch_next_observations, deterministic=False, reparameterize=False
            )
            q_next_target = torch.min(
                self.target_q1(batch_next_observations, next_actions),
                self.target_q2(batch_next_observations, next_actions),
            )
            if not self.deterministic_backup:
                alpha = self.log_alpha[0].exp()
                q_next_target -= alpha * next_log_pi

            q_target = (
                self.reward_scale * batch_rewards
                + (1 - batch_dones) * self.discount * q_next_target
            ).detach()
        # Bellman loss
        q1_pred = self.q1(batch_observations, batch_actions.float())
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_pred = self.q2(batch_observations, batch_actions.float())
        q2_loss = F.mse_loss(q2_pred, q_target)
        return q1_loss, q2_loss

    def compute_actor_and_alpha_loss(
        self, batch, alpha_optimizer, optimize: bool = True, log_type: str = "train"
    ):
        batch_observations, batch_actions = batch[:2]
        curr_actions, curr_log_pi = self.actor.get_actions(
            batch_observations, deterministic=False, reparameterize=True
        )

        alpha_loss = -(
            self.log_alpha[0] * (curr_log_pi + self.target_entropy).detach()
        ).mean()

        if optimize:
            alpha_optimizer.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_optimizer.step()

        alpha = self.log_alpha[0].exp()
        self.log(f"{log_type}/alpha", alpha, on_step=True)

        if self.current_epoch < self.bc_epochs:
            policy_log_prob = self.actor.log_prob(batch_observations, batch_actions)
            actor_loss = (alpha * curr_log_pi - policy_log_prob).mean()
        else:
            q1 = self.q1(batch_observations, curr_actions)
            q2 = self.q2(batch_observations, curr_actions)
            Q_value = torch.min(q1, q2)
            actor_loss = (alpha * curr_log_pi - Q_value).mean()

        return actor_loss, alpha_loss

    def populate_parallel(
        self, steps: int = 1000, strategy: str = "random", num_parallel_envs: int = 2
    ):
        """
        Carries out several random steps through multiple
        environments run in parallel to fill up the replay
        buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
            nproc: number of proccesses to use to fill the replay
                buffer in parallel
        """
        # Create parallel envs
        envs = [
            make_env_fn(self.env_cfg, rank, seed=self.env_cfg.seed)
            for rank in range(num_parallel_envs)
        ]
        envs = SubprocVecEnv(envs)

        # Perform random steps to populate the replay buffer
        step = 0
        observations = envs.reset()
        pbar = tqdm(total=steps)
        while step < steps:
            transf_obs = self.transform_manager(
                observations, transf_type="train", device=self.device
            )
            actions = self.agent.get_actions(
                actor=self.actor, observation=transf_obs, strategy=strategy
            )
            next_observations, rewards, dones, infos = envs.step(actions)

            # Fill replay buffer
            obs_list = (
                dict_to_list_of_dicts(observations)
                if isinstance(observations, dict)
                else observations
            )
            next_obs_list = (
                dict_to_list_of_dicts(next_observations)
                if isinstance(next_observations, dict)
                else next_observations
            )
            for i, done in enumerate(dones):
                next_obs = (
                    infos[i]["terminal_observation"] if done else next_obs_list[i]
                )
                self.replay_buffer.add_transition(
                    obs_list[i], actions[i], next_obs, rewards[i], dones[i]
                )
                step += 1
                pbar.update(1)
            observations = next_observations
        pbar.close()

    def populate(
        self, steps: int = 1000, strategy: str = "random", num_parallel_envs: int = 1
    ) -> None:
        """
        Carries out several steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
            strategy: strategy to follow to select actions to fill the replay buffer
            fill_parallel: boolean determining if the replay buffer is
                filled sequentially or in a parallel manner
        """

        self.cons_logger.info("Populating replay buffer with warm up steps")
        if num_parallel_envs > 1:
            self.populate_parallel(steps, strategy, num_parallel_envs=num_parallel_envs)
        else:
            for _ in tqdm(range(steps)):
                self.agent.play_step(self.actor, strategy, self.device)
        self.replay_buffer.save(self.replay_buffer_path)

    def log_train_metrics(self):
        self.log("train/avg_reward", np.mean(self.rewards), on_step=True)
        self.log("train/accuracy", np.mean(self.accuracies), on_step=True)
        self.log(
            "train/avg_episode_return", np.mean(self.episodes_returns), on_step=True
        )
        self.log(
            "train/avg_episode_length", np.mean(self.episodes_lengths), on_step=True
        )
        self.log("train/episode_return", self.episode_return, on_step=True)
        self.log("train/episode_length", self.episode_length, on_step=True)
        self.log("train/episode_number", self.episode_number.item(), on_step=True)
        self.log("train/episode_step", self.episode_step.item(), on_step=True)
        self.cons_logger.info(
            "Training - Episode: %d, Return: %.2f"
            % (self.episode_number, self.episode_return)
        )

    def play_step(self):
        reward, self.episode_done, success = self.agent.play_step(
            self.actor, "stochastic", self.device
        )
        self.rewards.append(reward)
        self.episode_return += reward
        self.episode_length += 1.0

        if self.episode_done:
            self.accuracies.append(int(success))
            self.episode_number += 1.0
            self.episode_step += self.episode_length
            self.episodes_returns.append(self.episode_return)
            self.episodes_lengths.append(self.episode_length)
            self.log_train_metrics()
            self.episode_return, self.episode_length = 0.0, 0.0

    def overwrite_batch(self, batch):
        obs, actions, next_obs, rew, dones = batch
        # Verifying batch shape
        if len(rew.shape) == 1:
            rew = rew.unsqueeze(-1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(-1)

        # Verifying input type
        rew = rew.float()
        actions = actions.float()
        dones = dones.int()
        if not isinstance(obs, dict):
            obs = obs.float()
            next_obs = next_obs.float()

        # Verifying device
        if rew.device != self.device:
            rew = rew.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        batch = obs, actions, next_obs, rew, dones
        return batch

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        """
        self.play_step()
        batch = self.overwrite_batch(batch)
        self.compute_update(batch)

    def save_checkpoint(self):
        """Checkpoints used to restart training"""
        if self.model_dir is not None:
            self.trainer.save_checkpoint(self.model_dir / "last.ckpt")
        if self.replay_buffer is not None:
            self.replay_buffer.save(self.replay_buffer_path)

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        alpha_optimizer = optim.Adam([self.log_alpha], lr=self.actor_lr)
        actor_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()),
            lr=self.actor_lr,
        )
        q1_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.q1.parameters()),
            lr=self.critic_lr,
        )
        q2_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.q2.parameters()),
            lr=self.critic_lr,
        )
        return [alpha_optimizer, actor_optimizer, q1_optimizer, q2_optimizer]
