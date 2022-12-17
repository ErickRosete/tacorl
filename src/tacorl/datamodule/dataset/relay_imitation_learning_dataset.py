import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from tacorl.utils.transforms import TransformManager

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class RILDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        modalities: DictConfig,
        max_low_level_window: int = 30,
        max_high_level_window: int = 260,
        action_type: str = "rel_actions_world",
        train: bool = True,
        transf_type: str = "train",
        n_digits: Optional[int] = None,
        transform_manager: TransformManager = None,
    ):
        self.max_low_level_window = max_low_level_window
        self.max_high_level_window = max_high_level_window

        self.action_type = action_type
        assert action_type in modalities, f"{action_type} need to be present in dataset"
        self.modalities = modalities

        self.train = train
        self.transf_type = transf_type

        self.abs_datasets_dir = data_dir
        assert (
            self.abs_datasets_dir.is_dir()
        ), f"{str(self.abs_datasets_dir)} is not a dir"

        self.episode_lookup: List[int] = []
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")
        (
            self.episode_lookup,
            self.max_batched_length_per_demo,
        ) = self.load_file_indices()
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern(n_digits)
        self.transform_manager = transform_manager

    def __len__(self) -> int:
        """
        returns
        ----------
        number of possible starting frames
        """
        return len(self.episode_lookup)

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def sample_goal_step(self, start_step, end_step):
        step_options = np.arange(start_step, end_step)
        if end_step <= start_step:
            return end_step
        return np.random.choice(step_options)

    def get_transitions(self, step: int):
        ep_end = self.find_episode_end(step)

        # Low level transition
        low_level_max_end_step = min(ep_end, step + self.max_low_level_window)
        low_level_goal_step = self.sample_goal_step(step + 1, low_level_max_end_step)
        obs = self.get_file_from_step(step, transform=True)
        action = obs.pop(self.action_type, None)
        low_level_goal = self.get_file_from_step(low_level_goal_step, transform=True)
        low_level_goal.pop(self.action_type, None)

        # High level transition
        # low_level_max_end_step will be used as subgoal
        high_level_max_end_step = min(ep_end, step + self.max_high_level_window)
        high_level_goal_step = self.sample_goal_step(
            low_level_max_end_step, high_level_max_end_step
        )
        high_level_goal = self.get_file_from_step(high_level_goal_step, transform=True)
        high_level_goal.pop(self.action_type, None)
        subgoal = self.get_file_from_step(low_level_max_end_step, transform=True)
        subgoal.pop(self.action_type, None)

        transition = {
            "obs": obs,
            "low_level_goal": low_level_goal,
            "low_level_action": action,
            "high_level_goal": high_level_goal,
            "high_level_action": subgoal,
        }
        return transition

    def __getitem__(self, idx: int) -> Dict:
        step = self.episode_lookup[idx]
        transitions = self.get_transitions(step)
        return transitions

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def get_file_from_step(self, step, transform=True):
        file_path = self.get_frame_name(step)
        data = load_npz(file_path)
        filtered_data = {modality: data[modality] for modality in self.modalities}
        if transform:
            return self.transform_manager(filtered_data, transf_type=self.transf_type)
        return filtered_data

    def lookup_naming_pattern(self, n_digits):
        it = os.scandir(self.abs_datasets_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = (
            n_digits
            if n_digits is not None
            else len(re.findall(r"\d+", filename.stem)[0])
        )
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_frame_name(self, step: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_pattern[0]}{step:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def set_ep_start_end_ids(self):
        """Extracts ep_start_end_ids from the adequate partition stored in split.json,
        if it doesn't have this file it searches for ep_start_end_ids.npy"""

        if (self.abs_datasets_dir / "split.json").is_file():
            with open(self.abs_datasets_dir / "split.json") as f:
                data_split = json.load(f)
            split_keys = list(data_split.keys())
            train_key = [key for key in split_keys if "train" in key][0]
            val_key = [key for key in split_keys if "val" in key][0]
            split_key = train_key if self.train else val_key
            assert (
                split_key in data_split
            ), f"data split file doesn't contain {split_key} key"
            self.ep_start_end_ids = np.array(data_split[split_key])
            logger.info(
                f"Found 'split.json' with {len(self.ep_start_end_ids)} episodes."
            )
        else:
            self.ep_start_end_ids = np.load(
                self.abs_datasets_dir / "ep_start_end_ids.npy"
            )
            logger.info(
                f"Found 'ep_start_end_ids.npy' with {len(self.ep_start_end_ids)} "
                "episodes."
            )

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
            assert end_idx > self.max_low_level_window
            for idx in range(start_idx, end_idx + 1 - self.max_low_level_window):
                episode_lookup.append(idx)
            possible_indices = end_idx + 1 - start_idx - self.max_low_level_window
            max_batched_length_per_demo.append(possible_indices)
        return episode_lookup, max_batched_length_per_demo
