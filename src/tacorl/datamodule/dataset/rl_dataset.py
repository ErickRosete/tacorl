# import time
from typing import Iterator

from torch.utils.data.dataset import IterableDataset

from tacorl.modules.sac.replay_buffer import ReplayBuffer
from tacorl.utils.misc import dict_to_list_of_dicts, list_of_dicts_to_dict
from tacorl.utils.transforms import TransformManager


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 64,
        transform_manager: TransformManager = {},
        device: str = "cuda",
    ) -> None:
        """
        Args:
            replay_buffer: replay buffer storing env transitions
            sample_size: number of experiences to sample at a time
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.transform_manager = transform_manager
        self.device = device

    def transform_state(self, states):
        if isinstance(states[0], dict) and "goal" in states[0]:
            states = list_of_dicts_to_dict(states)
            transf_obs = self.apply_transforms(states["observation"])
            transf_goal = self.apply_transforms(states["goal"])
            transf_state = [
                {"observation": transf_obs[i], "goal": transf_goal[i]}
                for i in range(len(transf_obs))
            ]
        else:
            transf_state = self.apply_transforms(states)
        return transf_state

    def apply_transforms(self, states: list):
        if isinstance(states[0], dict):
            states = list_of_dicts_to_dict(states, to_numpy=True)
            states = self.transform_manager(states, "train", device=self.device)
            return dict_to_list_of_dicts(states)
        return self.transform_manager(states, "train", device=self.device)

    def __iter__(self) -> Iterator:
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = self.transform_state(states)
        next_states = self.transform_state(next_states)
        for i in range(len(dones)):
            yield states[i], actions[i], next_states[i], rewards[i], dones[i]
