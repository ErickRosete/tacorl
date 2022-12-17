from typing import List, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Independent, Normal
from torch.optim.optimizer import Optimizer

from tacorl.modules.sac.sac_lightning import SAC
from tacorl.utils.misc import expand_obs
from tacorl.utils.networks import get_batch_size_from_input


class CQL_Online(SAC):
    """Basic Online Conservative Q Learning (CQL)
    implementation using PyTorch Lightning"""

    def __init__(
        self,
        conservative_weight: float = 1.0,
        lagrange_thresh: float = 5.0,
        n_action_samples: int = 10,
        temp: float = 1.0,
        with_lagrange: bool = False,
        with_dr3: bool = False,
        dr3_coefficient: float = 0.03,
        with_vib: bool = False,
        vib_coefficient: float = 0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.conservative_weight = conservative_weight
        self.n_action_samples = n_action_samples
        self.temp = temp
        self.with_lagrange = with_lagrange
        self.with_dr3 = with_dr3
        self.dr3_coefficient = dr3_coefficient
        self.with_vib = with_vib
        self.vib_coefficient = vib_coefficient
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = nn.Parameter(torch.zeros(1), requires_grad=True)

    def compute_random_values(self, obs):
        batch_size = get_batch_size_from_input(obs)
        expanded_obs = expand_obs(obs, self.n_action_samples)

        # Sample random uniform actions
        action_dim = self.actor.action_dim
        flat_shape = (batch_size * self.n_action_samples, action_dim)
        zero_tensor = torch.zeros(flat_shape, device=self.device)
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        if self.actor.discrete_gripper:
            random_actions[..., -1] = torch.where(
                random_actions[..., -1] >= 0, 1.0, -1.0
            )

        # Get Q values of the uniform actions
        q1 = self.q1(expanded_obs, random_actions)
        q2 = self.q2(expanded_obs, random_actions)
        # (n * bs, 1) -> (n, bs) -> (bs, n)
        q1 = q1.view(self.n_action_samples, batch_size).transpose(0, 1)
        q2 = q2.view(self.n_action_samples, batch_size).transpose(0, 1)
        random_log_probs = np.log(0.5 ** action_dim)
        return q1, q2, random_log_probs

    def compute_policy_values(self, policy_obs, value_obs):
        """Compute Q values for value_obs and
        actions obtained from policy_obs"""
        with torch.no_grad():
            n_actions, n_log_pi = self.actor.sample_n_with_log_prob(
                policy_obs, n_actions=self.n_action_samples
            )

        batch_size = get_batch_size_from_input(value_obs)
        expanded_value_obs = expand_obs(value_obs, self.n_action_samples)
        flat_actions = n_actions.reshape(-1, n_actions.shape[-1])  # n * bs, act_dim

        q1 = self.q1(expanded_value_obs, flat_actions)
        q2 = self.q2(expanded_value_obs, flat_actions)
        # (n * bs, 1) -> (n, bs) -> (bs, n)
        q1 = q1.view(self.n_action_samples, batch_size).transpose(0, 1)
        q2 = q2.view(self.n_action_samples, batch_size).transpose(0, 1)
        # (n, bs, 1) -> (bs, n)
        log_probs = n_log_pi.squeeze(-1).transpose(0, 1)
        return q1, q2, log_probs

    def compute_conservative_loss(
        self, batch, optimize: bool = True, log_type: str = "train"
    ):
        (
            batch_observations,
            batch_actions,
            batch_next_observations,
            batch_rewards,
            batch_dones,
        ) = batch

        # Calculate Q values
        q1_rand, q2_rand, random_density = self.compute_random_values(
            batch_observations
        )
        q1_curr_actions, q2_curr_actions, curr_log_pis = self.compute_policy_values(
            batch_observations, batch_observations
        )
        q1_next_actions, q2_next_actions, next_log_pis = self.compute_policy_values(
            batch_next_observations, batch_observations
        )
        q1_data = self.q1(batch_observations, batch_actions)
        q2_data = self.q2(batch_observations, batch_actions)

        # Log q values mean for debugging
        self.log(f"{log_type}/q1_data", q1_data.mean(), on_step=True)
        self.log(f"{log_type}/q1_random", q1_rand.mean(), on_step=True)
        self.log(
            f"{log_type}/q1_policy",
            q1_curr_actions.mean(),
            on_step=True,
        )
        self.log(f"{log_type}/q2_data", q2_data.mean(), on_step=True)
        self.log(f"{log_type}/q2_random", q2_rand.mean(), on_step=True)
        self.log(
            f"{log_type}/q2_policy",
            q2_curr_actions.mean(),
            on_step=True,
        )

        # bs, 3 * num_actions
        cat_q1 = torch.cat(
            [
                q1_rand - random_density,
                q1_curr_actions - curr_log_pis.detach(),
                q1_next_actions - next_log_pis.detach(),
            ],
            dim=1,
        )
        cat_q2 = torch.cat(
            [
                q2_rand - random_density,
                q2_curr_actions - curr_log_pis.detach(),
                q2_next_actions - next_log_pis.detach(),
            ],
            dim=1,
        )

        cons_q1_loss = (
            torch.logsumexp(cat_q1 / self.temp, dim=1).mean()
            * self.conservative_weight
            * self.temp
        )
        cons_q2_loss = (
            torch.logsumexp(cat_q2 / self.temp, dim=1).mean()
            * self.conservative_weight
            * self.temp
        )

        # Maximize Q-values under data distribution
        cons_q1_loss -= q1_data.mean() * self.conservative_weight
        cons_q2_loss -= q2_data.mean() * self.conservative_weight

        if self.with_lagrange:
            alpha_prime = torch.clamp(
                self.log_alpha_prime[0].exp(), min=0.0, max=1000000.0
            )
            self.log(f"{log_type}/alpha_prime", alpha_prime, on_step=True)

            cons_q1_loss = alpha_prime * (cons_q1_loss - self.target_action_gap)
            cons_q2_loss = alpha_prime * (cons_q2_loss - self.target_action_gap)
            alpha_prime_loss = (-cons_q1_loss - cons_q2_loss) * 0.5
            self.log(f"{log_type}/alpha_prime_loss", alpha_prime_loss, on_step=True)

            if optimize:
                alpha_prime_optimizer = self.optimizers()[4]
                alpha_prime_optimizer.zero_grad()
                self.manual_backward(alpha_prime_loss, retain_graph=True)
                alpha_prime_optimizer.step()

        return cons_q1_loss, cons_q2_loss

    def compute_network_vib_loss(self, batch, network):
        batch_observations = batch[0]
        vib_dist = network.get_vib_distribution(batch_observations)
        prior = Independent(
            Normal(torch.zeros_like(vib_dist.mean), torch.ones_like(vib_dist.stddev)), 1
        )
        vib_loss = self.vib_coefficient * D.kl_divergence(vib_dist, prior).mean()
        return vib_loss

    def compute_vib_loss(self, batch, log_type: str = "train"):
        q1_vib_loss = self.compute_network_vib_loss(batch, network=self.q1)
        q2_vib_loss = self.compute_network_vib_loss(batch, network=self.q2)
        self.log(f"{log_type}/q1_vib_loss", q1_vib_loss, on_step=True)
        self.log(f"{log_type}/q2_vib_loss", q2_vib_loss, on_step=True)
        return q1_vib_loss, q2_vib_loss

    def compute_network_dr3_loss(self, batch, network):
        batch_observations, _, batch_next_observations = batch[:3]
        emb_obs = network.get_emb_obs_representation(batch_observations)
        emb_next_obs = network.get_emb_obs_representation(batch_next_observations)
        dr3_loss = (emb_obs * emb_next_obs.detach()).sum(dim=1).mean(dim=0)
        dr3_loss *= self.dr3_coefficient
        return dr3_loss

    def compute_dr3_loss(self, batch, log_type: str = "train"):
        q1_dr3_loss = self.compute_network_dr3_loss(batch=batch, network=self.q1)
        q2_dr3_loss = self.compute_network_dr3_loss(batch=batch, network=self.q2)
        self.log(f"{log_type}/q1_dr3_loss", q1_dr3_loss, on_step=True)
        self.log(f"{log_type}/q2_dr3_loss", q2_dr3_loss, on_step=True)
        return q1_dr3_loss, q2_dr3_loss

    def compute_update(
        self,
        batch,
        optimize: bool = True,
        log_type: str = "train",
    ):
        (
            alpha_optimizer,
            actor_optimizer,
            q1_optimizer,
            q2_optimizer,
        ) = self.optimizers()[:4]

        # Actor
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(
            batch, alpha_optimizer, optimize=optimize, log_type=log_type
        )

        # Critic
        bellman_q1_loss, bellman_q2_loss = self.compute_critic_loss(batch)
        cons_q1_loss, cons_q2_loss = self.compute_conservative_loss(
            batch, optimize=optimize, log_type=log_type
        )
        q1_loss = bellman_q1_loss + cons_q1_loss
        q2_loss = bellman_q2_loss + cons_q2_loss

        if self.with_dr3:
            q1_dr3_loss, q2_dr3_loss = self.compute_dr3_loss(batch, log_type=log_type)
            q1_loss += q1_dr3_loss
            q2_loss += q2_dr3_loss

        if self.with_vib:
            q1_vib_loss, q2_vib_loss = self.compute_vib_loss(batch, log_type=log_type)
            q1_loss += q1_vib_loss
            q2_loss += q2_vib_loss

        # Log values
        self.log(f"{log_type}/bellman_q1_loss", bellman_q1_loss, on_step=True)
        self.log(f"{log_type}/conservative_q1_loss", cons_q1_loss, on_step=True)
        self.log(f"{log_type}/q1_loss", q1_loss, on_step=True)

        self.log(f"{log_type}/bellman_q2_loss", bellman_q2_loss, on_step=True)
        self.log(f"{log_type}/conservative_q2_loss", cons_q2_loss, on_step=True)
        self.log(f"{log_type}/q2_loss", q2_loss, on_step=True)

        self.log(f"{log_type}/actor_loss", actor_loss, on_step=True)
        self.log(f"{log_type}/alpha_loss", alpha_loss, on_step=True)

        # Optimize
        if optimize:
            actor_optimizer.zero_grad()
            self.manual_backward(actor_loss, retain_graph=True)
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.clip_grad_val
                )
            actor_optimizer.step()

            q1_optimizer.zero_grad()
            self.manual_backward(q1_loss, retain_graph=True)
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.clip_grad_val)
            q1_optimizer.step()

            q2_optimizer.zero_grad()
            self.manual_backward(q2_loss)
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.clip_grad_val)
            q2_optimizer.step()

            # Soft updates
            self.soft_update_from_to(self.q1, self.target_q1, self.tau)
            self.soft_update_from_to(self.q2, self.target_q2, self.tau)

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
        self.compute_update(batch, optimize=True, log_type="train")

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize optimizers"""
        optimizers = super().configure_optimizers()
        if self.with_lagrange:
            alpha_prime_optimizer = optim.Adam(
                [self.log_alpha_prime], lr=self.critic_lr
            )
            optimizers.append(alpha_prime_optimizer)
        return optimizers
