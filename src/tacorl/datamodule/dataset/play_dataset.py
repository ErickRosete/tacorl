import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from tacorl.utils.episode_utils import get_state_info_dict
from tacorl.utils.transforms import TransformManager

logger = logging.getLogger(__name__)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def get_validation_window_size(idx, min_window_size, max_window_size):
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hash(str(idx)) % window_range


class PlayDataset(Dataset):
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
        data_dir: Path,
        modalities: DictConfig,
        action_type: str = "rel_actions_world",
        train: bool = True,
        real_world: bool = False,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        transform_manager: TransformManager = None,
        transf_type: str = "train",
        skip_frames: int = 0,
        n_digits: Optional[int] = None,
        include_goal: bool = False,
        goal_augmentation: bool = False,
        goal_sampling_prob: float = 0.3,
        goal_strategy_prob: dict = {
            "geometric": 0.5,
            "similar_robot_obs": 0.5,
        },
        nn_steps_from_step_path: str = "nn_steps_from_step.pkl",
        num_nn: int = 32,
    ):
        self.action_type = action_type
        assert action_type in modalities, f"{action_type} need to be present in dataset"
        self.real_world = real_world
        self.modalities = modalities

        # Scene obs are not available in real world data
        if self.real_world and "scene_obs" in self.modalities:
            self.modalities.remove("scene_obs")

        self.pad = pad
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = data_dir
        assert (
            self.abs_datasets_dir.is_dir()
        ), f"{str(self.abs_datasets_dir)} is not a dir"

        self.train = train
        self.episode_lookup: List[int] = []
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")
        self.transform_manager = transform_manager
        self.transf_type = transf_type
        self.skip_frames = skip_frames
        (
            self.episode_lookup,
            self.max_batched_length_per_demo,
        ) = self.load_file_indices()
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern(n_digits)
        self.include_goal = include_goal
        if self.include_goal:
            self.set_nn_steps_from_step(nn_steps_from_step_path, num_nn=num_nn)
        self.goal_sampling_prob = goal_sampling_prob
        self.goal_augmentation = goal_augmentation
        if include_goal:
            assert np.isclose(
                np.sum(list(goal_strategy_prob.values())), 1.0
            ), "Goal strategy probability must sum to 1"
            self.goal_strategy_prob = goal_strategy_prob

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
                if self.train:
                    window_size = np.random.randint(
                        self.min_window_size, self.max_window_size + 1
                    )
                else:
                    window_size = get_validation_window_size(
                        idx, self.min_window_size, self.max_window_size
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

        # Rearrange sequence
        states = {
            modality: sequence[modality]
            for modality in self.modalities
            if "action" not in modality
        }
        actions = sequence[self.action_type]

        seq = {
            "states": states,
            "actions": actions,
            "idx": sequence["idx"],
            "window_size": sequence["window_size"],
        }
        if not self.real_world:
            seq["state_info"] = sequence["state_info"]

        if self.include_goal:
            goal_strategy = self.sample_goal_strategy()
            if goal_strategy == "geometric":
                seq["goal"], seq["disp"] = self.get_future_state(idx, window_size)
            elif goal_strategy == "similar_robot_obs":
                seq_start = self.episode_lookup[idx]
                seq["goal"] = self.get_similar_robot_obs_state(
                    seq_start + window_size - 1
                )
                seq["disp"] = -1
        return seq

    def sample_goal_strategy(self):
        options = list(self.goal_strategy_prob.keys())
        prob = list(self.goal_strategy_prob.values())
        return np.random.choice(options, p=prob)

    def get_similar_robot_obs_state(self, step):
        step_options = self.nn_steps_from_step[step]
        if len(step_options) == 0:
            return self.get_random_state()
        goal_step = np.random.choice(step_options)
        return self.get_file(goal_step, transform=True)

    def set_nn_steps_from_step(
        self,
        nn_steps_from_step_path,
        num_nn: int = 32,
        margin: int = 16,
    ):
        """Find steps with similar robot configuration"""
        nn_steps_from_step = {}
        nn_steps_from_step_path = Path(nn_steps_from_step_path)
        nn_steps_from_step_path = nn_steps_from_step_path.expanduser()
        if nn_steps_from_step_path.is_file():
            with open(nn_steps_from_step_path) as f:
                nn_steps_from_step = json.load(f)

        data_type = "train" if self.train else "validation"
        if data_type in nn_steps_from_step:
            self.nn_steps_from_step = {
                int(k): v for k, v in nn_steps_from_step[data_type].items()
            }  # maps str -> int
            return

        logger.info(f"Building nn_steps_from_step for {data_type}")

        import faiss  # Only required to build nn_steps_from_step

        step_robot_obs = OrderedDict()
        for start_step, end_step in tqdm(self.ep_start_end_ids):
            for step in tqdm(range(start_step, end_step)):
                file_path = self.get_episode_name(step)
                data = load_npz(file_path)
                step_robot_obs[step] = data["robot_obs"]

        robot_obs = np.stack(step_robot_obs.values()).astype("float32")
        index = faiss.IndexFlatL2(robot_obs.shape[-1])
        index.add(robot_obs)
        _, all_nn_indices = index.search(robot_obs, num_nn)

        new_nn_steps_from_step = {}
        steps = list(step_robot_obs.keys())
        for query_idx, nn_indices in enumerate(all_nn_indices):
            query_step = steps[query_idx]
            new_nn_steps_from_step[query_step] = []
            for index in nn_indices:
                nn_step = steps[index]
                if not (nn_step + margin > query_step > nn_step - margin):
                    new_nn_steps_from_step[query_step].append(nn_step)

        logger.info("Successfully built nn_steps_from_step")
        nn_steps_from_step[data_type] = new_nn_steps_from_step
        with open(nn_steps_from_step_path, "w") as f:
            json.dump(nn_steps_from_step, f, indent=4)
        self.nn_steps_from_step = nn_steps_from_step[data_type]

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def get_file(self, file_idx, transform=True):
        file_path = self.get_episode_name(file_idx)
        data = load_npz(file_path)
        filtered_data = {
            modality: data[modality]
            for modality in self.modalities
            if "action" not in modality
        }
        if transform:
            return self.transform_manager(filtered_data, transf_type=self.transf_type)
        return filtered_data

    def get_random_state(self):
        file_idx = np.random.choice(self.episode_lookup)
        return self.get_file(file_idx, transform=True)

    def get_future_state(self, idx, window_size):
        """Tries to return a random future state between
        [seq_end, episode_end] (Low and high inclusive)
        following the geometric distribution"""

        seq_start = self.episode_lookup[idx]
        # init_disp_step = seq_start + window_size - 2

        episode_end = self.find_episode_end(seq_start)
        if episode_end is None:
            return self.get_random_state()
        disp = np.random.default_rng().geometric(p=self.goal_sampling_prob)
        goal_step = seq_start + (window_size - 1) * disp
        if self.goal_augmentation:
            noise_step = np.random.randint(3) - 1
            goal_step += noise_step
        file_step = min(episode_end, goal_step)
        future_state = self.get_file(file_step, transform=True)
        return future_state, disp

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
        for modality in self.modalities:
            if "rel" in modality:
                # Zero pad all and repeat the gripper action
                seq[modality] = torch.cat(
                    [
                        self.pad_with_zeros(seq[modality][..., :-1], pad_size),
                        self.pad_with_repetition(seq[modality][..., -1:], pad_size),
                    ],
                    dim=-1,
                )
            else:
                seq[modality] = self.pad_with_repetition(seq[modality], pad_size)

        # Update state info
        if not self.real_world:
            seq["state_info"] = {
                k: self.pad_with_repetition(v, pad_size)
                for k, v in seq["state_info"].items()
            }

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

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def zip_sequence(
        self, start_idx: int, end_idx: int, idx: int
    ) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files
        and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame

        returns:
        -----------
        episode: dict of numpy arrays containing the episode
                 where keys are the names of modalities
        """
        episodes = [
            load_npz(self.get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        keys_to_load = self.modalities.copy()

        if not self.real_world:
            if "robot_obs" not in keys_to_load:
                keys_to_load.append("robot_obs")
            if "scene_obs" not in keys_to_load:
                keys_to_load.append("scene_obs")

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys_to_load}
        return episode

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

        seq = self.zip_sequence(start_file_indx, end_file_indx, idx)

        if not self.real_world:
            info = get_state_info_dict(seq)
        # Filter modalities
        seq = {modality: seq[modality] for modality in self.modalities}
        # Apply transformations
        seq = self.transform_manager(seq, transf_type=self.transf_type)
        # Add info
        if not self.real_world:
            seq.update(info)
        seq["idx"] = idx
        seq["window_size"] = window_size
        return seq

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
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.max_window_size):
                episode_lookup.append(idx)
            possible_indices = end_idx + 1 - start_idx - self.max_window_size
            max_batched_length_per_demo.append(possible_indices)
        return episode_lookup, max_batched_length_per_demo
