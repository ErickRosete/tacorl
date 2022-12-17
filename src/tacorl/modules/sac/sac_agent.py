import logging

import gym
import numpy as np
import torch

from tacorl.modules.sac.replay_buffer import ReplayBuffer
from tacorl.utils.networks import get_batch_size_from_input
from tacorl.utils.transforms import TransformManager, transform_observation


class SAC_Agent:
    """
    Base Agent class handling the interaction with the environment
    """

    def __init__(
        self,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        transform_manager: TransformManager,
    ) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.replay_buffer = replay_buffer
        self.transform_manager = transform_manager
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state"""
        self.observation = self.env.reset()

    @torch.no_grad()
    def play_step(self, actor, strategy="stochastic", device="cuda"):
        """Perform a step in the environment and add the transition
        tuple to the replay buffer"""

        transf_obs = transform_observation(
            transform_manager=self.transform_manager,
            obs=self.observation,
            transf_type="train",
            device=device,
        )
        action = self.get_actions(actor, transf_obs, strategy)
        next_observation, reward, done, info = self.env.step(action)
        self.replay_buffer.add_transition(
            self.observation, action, next_observation, reward, done
        )
        self.observation = next_observation
        success = False
        if done:
            self.reset()
            success = ("success" in info) and info["success"]
        return reward, done, success

    def get_actions(self, actor, observation, strategy="stochastic"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        if strategy == "stochastic":
            actions, _ = actor.get_actions(
                observation, deterministic=False, reparameterize=False
            )
            return actions.detach().cpu().numpy()
        elif strategy == "deterministic":
            actions, _ = actor.get_actions(
                observation, deterministic=True, reparameterize=False
            )
            return actions.detach().cpu().numpy()
        else:
            batch_size = get_batch_size_from_input(observation)
            if strategy == "random":
                actions = [self.env.action_space.sample() for _ in range(batch_size)]
                return np.squeeze(np.stack(actions))
            elif strategy == "zeros":
                return np.squeeze(np.zeros((batch_size, *self.env.action_space.shape)))
            else:
                raise Exception("Strategy not implemented")
