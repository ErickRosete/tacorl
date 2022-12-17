import logging
from typing import Dict, List, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import torch
from torch.utils.data import Dataset

from tacorl.utils.transforms import TransformManager

logger = logging.getLogger(__name__)


class D4RLPlayDataset(Dataset):
    """
    Dataset Loader that uses a shared memory cache

    parameters
    ----------

    data_dir:           path of folder containing episode files
    save_format:        format of episodes in datasets_dir (.pkl or .npz)
    obs_space:          DictConfig of the observation modalities of the dataset
    max_window_size:    maximum length of the episodes sampled from the dataset
    """

    def __init__(
        self,
        min_window_size: int = 8,
        max_window_size: int = 16,
        pad: bool = True,
        transform_manager: TransformManager = None,
        transf_type: str = "train",
        include_goal: bool = False,
        goal_sampling_prob: float = 0.3,
        d4rl_env: str = "antmaze-large-diverse-v0",
        goal_augmentation: bool = False,
        goal_threshold: float = 0.5,
    ):

        env = gym.make(d4rl_env)
        self.dataset = env.get_dataset()

        self.pad = pad
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

        self.episode_lookup: List[int] = []
        self.transform_manager = transform_manager
        self.goal_augmentation = goal_augmentation
        self.transf_type = transf_type
        (
            self.episode_lookup,
            self.max_batched_length_per_demo,
        ) = self.load_file_indices()
        self.include_goal = include_goal
        self.goal_sampling_prob = goal_sampling_prob
        self.goal_threshold = goal_threshold

    def __len__(self) -> int:
        """
        returns
        ----------
        number of possible starting frames
        """
        return len(self.episode_lookup)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = np.random.randint(
                    self.min_window_size, self.max_window_size + 1
                )
            else:
                logger.error(
                    f"min_window_size {self.min_window_size} "
                    f"> max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx

        sequence = self.get_sequences(idx, window_size)
        if self.pad:
            sequence = self.pad_sequence(sequence, window_size)

        if self.include_goal:
            sequence["goal"], sequence["goal_reached"] = self.get_future_goal(
                idx, window_size
            )
        return sequence

    def get_file(self, file_idx, transform=True):
        obs = {
            "observations": self.dataset["observations"][file_idx],
            "actions": self.dataset["actions"][file_idx],
        }
        if transform:
            return self.transform_manager(obs, transf_type=self.transf_type)
        return obs

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def get_random_state(self):
        file_idx = np.random.choice(self.episode_lookup)
        return self.get_file(file_idx, transform=True)

    def extract_goal_from_state(self, state):
        goal = state["observations"][:2]
        # Augmentation for the goal
        if self.goal_augmentation:
            goal += np.random.uniform(low=-0.1, high=0.1, size=2)
        return goal

    def get_future_goal(self, idx, window_size):
        """Tries to return a random future state between
        [seq_end, episode_end] (Low and high inclusive)
        following the geometric distribution"""

        seq_start = self.episode_lookup[idx]
        episode_end = self.find_episode_end(seq_start)
        if episode_end is None:
            goal_state = self.get_random_state()
            goal = self.extract_goal_from_state(goal_state)
        else:
            disp = np.random.default_rng().geometric(p=self.goal_sampling_prob)
            goal_step = seq_start + (window_size - 1) * disp
            if self.goal_augmentation:
                noise_step = np.random.randint(3) - 1
                goal_step += noise_step
            file_step = min(episode_end, goal_step)
            goal_state = self.get_file(file_step, transform=True)
            goal = self.extract_goal_from_state(goal_state)

        seq_end_pos = self.dataset["observations"][seq_start + window_size - 1][:2]
        reached = np.linalg.norm(goal - seq_end_pos) < self.goal_threshold
        return goal, reached

    @property
    def is_varying(self) -> bool:
        return self.min_window_size != self.max_window_size and not self.pad

    def pad_sequence(
        self,
        seq: Dict,
        window_size: int,
    ) -> Dict:

        # Update modalities
        pad_size = self.max_window_size - window_size
        # Zero pad all and repeat the gripper action
        seq["actions"] = self.pad_with_zeros(seq["actions"], pad_size)
        seq["observations"] = self.pad_with_repetition(seq["observations"], pad_size)
        return seq

    @staticmethod
    def pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """repeats the last element with pad_size"""
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        window_size:    length of sampled episode

        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """

        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + window_size
        actions = self.dataset["actions"][start_file_indx:end_file_indx]
        observations = self.dataset["observations"][start_file_indx:end_file_indx]
        seq = {"actions": actions, "observations": observations}
        # Apply transformations
        seq = self.transform_manager(seq, transf_type=self.transf_type)
        # Add info
        seq["idx"] = idx
        seq["window_size"] = window_size
        return seq

    def set_ep_start_end_ids(self):
        timeouts = self.dataset["timeouts"].nonzero()[0]
        terminals = self.dataset["terminals"].nonzero()[0]
        episode_ends = list(set(timeouts.tolist() + terminals.tolist()))
        episode_ends.sort()

        ep_start_end_ids = []
        start = 0
        for ep_end in episode_ends:
            if ep_end - start > self.min_window_size:
                ep_start_end_ids.append([start, ep_end])
            start = ep_end + 1
        self.ep_start_end_ids = ep_start_end_ids

    def load_file_indices(self) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used
        for loading the episodes

        parameters
        ----------
        abs_datasets_dir: absolute path of the directory containing the dataset

        returns
        ----------
        episode_lookup: list for the mapping from training example index
                        to episode (file) index
        max_batched_length_per_demo: list of possible starting indices per episode
        """
        self.set_ep_start_end_ids()

        episode_lookup = []
        max_batched_length_per_demo = []
        for start_idx, end_idx in self.ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.max_window_size):
                episode_lookup.append(idx)
            possible_indices = end_idx + 1 - start_idx - self.max_window_size
            max_batched_length_per_demo.append(possible_indices)
        return episode_lookup, max_batched_length_per_demo
