from typing import List, Optional, Union

import torch
import torch.nn as nn


class VisualActorWrapper(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        encoder: nn.Module,
        goal_encoder: Optional[nn.Module] = None,
        env_modalities: List[str] = [],
        goal_modalities: List[str] = [],
    ):

        super(VisualActorWrapper, self).__init__()
        self.actor = actor
        self.action_dim = actor.action_dim
        self.discrete_gripper = actor.discrete_gripper
        self.encoder = encoder
        self.goal_encoder = goal_encoder
        self.env_modalities = env_modalities
        self.goal_modalities = goal_modalities

    def get_emb_obs_representation(self, obs: Union[dict, torch.Tensor]):
        """Get emb representation only of the observation without the goal"""
        if not isinstance(obs, dict):
            return obs
        if len(self.goal_modalities) > 0 and "goal" in obs:
            obs_dict = obs["observation"]
        else:
            obs_dict = obs

        emb_representation = self.encoder.get_state_from_observation(
            observation=obs_dict,
            modalities=self.env_modalities,
        )
        return emb_representation

    def get_emb_representation(self, obs: Union[dict, torch.Tensor]):
        if not isinstance(obs, dict):
            return obs

        if len(self.goal_modalities) > 0 and "goal" in obs:
            emb_obs = self.encoder.get_state_from_observation(
                observation=obs["observation"],
                modalities=self.env_modalities,
            )
            emb_goal = self.encoder.get_state_from_observation(
                observation=obs["goal"],
                modalities=self.goal_modalities,
            )
            if self.goal_encoder is not None:
                emb_goal = self.goal_encoder(emb_goal)
            emb_representation = torch.cat([emb_obs, emb_goal], dim=-1)
        else:
            emb_representation = self.encoder.get_state_from_observation(
                observation=obs,
                modalities=self.env_modalities,
            )
        return emb_representation

    def forward(self, obs: Union[dict, torch.Tensor], *args, **kwargs):
        actor_input = self.get_emb_representation(obs=obs)
        return self.actor(actor_input, *args, **kwargs)

    def get_actions(self, observation: Union[dict, torch.Tensor], *args, **kwargs):
        actor_input = self.get_emb_representation(obs=observation)
        return self.actor.get_actions(actor_input, *args, **kwargs)

    def sample_n_with_log_prob(
        self, observation: Union[dict, torch.Tensor], *args, **kwargs
    ):
        actor_input = self.get_emb_representation(obs=observation)
        return self.actor.sample_n_with_log_prob(actor_input, *args, **kwargs)

    def log_prob(self, observation: Union[dict, torch.Tensor], *args, **kwargs):
        actor_input = self.get_emb_representation(obs=observation)
        return self.actor.log_prob(actor_input, *args, **kwargs)
