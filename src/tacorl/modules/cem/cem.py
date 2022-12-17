from typing import Tuple

import numpy as np
import torch
import torch.distributions as tdist

from tacorl.utils.misc import expand_obs, get_obs_device


class CEMOptimizer(object):
    """
    Implementation of the cross-entropy method to find the
    action that maximizes the Critic function
    """

    def __init__(
        self,
        q1,
        q2,
        batch_size: int = 256,
        num_iterations: int = 4,
        elite_fraction: float = 0.1,
        min_std: float = 1e-3,
        max_std: float = 0.3,
        alpha: float = 0.1,
        action_dim: int = 7,
        discrete_gripper: bool = False,
    ) -> None:

        """
        critic: Neural networks that maps from state to the
                expected discounted return of rewards
        batch_size: number of samples of theta to evaluate per batch
        num_iterations: number of batches
        elite_frac: each batch, select this fraction of the top-performing samples
        initial_std: initial mean over input distribution

        """
        self.q1 = q1
        self.q2 = q2
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.elite_fraction = elite_fraction
        self.min_std = min_std
        self.max_std = max_std
        self.alpha = alpha
        self.action_dim = action_dim
        self.discrete_gripper = discrete_gripper

    def initialize_population_parameters(
        self, initial_mean: torch.Tensor = None, device: torch.device = "cuda"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = (
            initial_mean
            if initial_mean is not None
            else torch.zeros(self.action_dim, device=device)
        )
        std = torch.ones(self.action_dim, device=device) * self.max_std
        return mean, std

    def update_population_parameters(
        self, elites: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.alpha * mean + (1 - self.alpha) * torch.mean(elites, axis=0)
        std = self.alpha * std + (1 - self.alpha) * torch.std(elites, axis=0)
        std = std.clamp(min=self.min_std, max=self.max_std)
        return mean, std

    def get_action(self, obs, initial_mean: torch.Tensor = None) -> torch.Tensor:
        mean, std = self.initialize_population_parameters(
            initial_mean=initial_mean, device=get_obs_device(obs)
        )
        n_elite = int(np.round(self.batch_size * self.elite_fraction))

        best_q_value = -float("inf")
        for _ in range(self.num_iterations):
            # Get action population
            actions = (
                tdist.Normal(
                    loc=mean,
                    scale=std,
                )
                .sample((self.batch_size,))
                .clamp(min=-1.0, max=1.0)
            )
            if self.discrete_gripper:
                actions[..., -1] = torch.where(actions[..., -1] >= 0, 1.0, -1.0)

            # Evaluate q value for action population
            expanded_obs = expand_obs(obs, self.batch_size, reshape=False)
            # Q_value = self.critic(expanded_obs, actions)
            q1_value = self.q1(expanded_obs, actions)
            q2_value = self.q1(expanded_obs, actions)
            Q_value = torch.min(q1_value, q2_value)
            elite_inds = torch.argsort(Q_value, dim=0, descending=True)[:n_elite, 0]
            elite_actions = actions[elite_inds]
            mean, std = self.update_population_parameters(
                elites=elite_actions, mean=mean, std=std
            )
            it_best_q_value = Q_value[elite_inds][0].item()
            if it_best_q_value > best_q_value:
                best_action = elite_actions[0]
                best_q_value = it_best_q_value
        return best_action
