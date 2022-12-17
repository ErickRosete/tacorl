import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from tacorl.utils.path import get_file_list
from tacorl.utils.transforms import TransformManager

logger = logging.getLogger(__name__)


class GoalCondReplayBufferDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        modalities: list,
        action_type: str = "rel_actions_world",
        transform_manager: TransformManager = None,
        train: bool = True,
        transf_type: str = "train",
        device: str = "cpu",
        goal_strategy_prob: dict = {
            "geometric": 0.5,
            "similar_robot_obs": 0.5,
        },
        initial_horizon: int = 8,
        horizon_step: int = 4,
        max_horizon: int = 256,
        nn_steps_from_step_path: str = "nn_steps_from_step.json",
        num_nn: int = 32,
        filter_by_tasks: bool = False,
        tasks: List[str] = [],
        goal_sampling_prob: float = 0.3,
        *args,
        **kwargs,
    ):
        """
        Args
            data_dir: path of the dir where the data is stored
            modalities: list of strings with the keys of the modalities that
                        will be used to train the network
        """
        self.action_type = action_type
        assert action_type in modalities, f"{action_type} need to be present in dataset"
        assert np.isclose(
            np.sum(list(goal_strategy_prob.values())), 1.0
        ), "Goal strategy probability must sum to 1"
        self.goal_strategy_prob = goal_strategy_prob
        self.data_dir = Path(data_dir).expanduser()
        self.modalities = modalities
        self.train = train
        self.transform_manager = transform_manager
        self.transf_type = transf_type
        self.device = device
        self.set_step_to_file()

        self.initial_horizon = initial_horizon
        self.current_horizon = initial_horizon
        self.horizon_step = horizon_step
        self.max_horizon = max_horizon
        if "task_future" in goal_strategy_prob.keys() or filter_by_tasks:
            self.set_lang_ann()
        self.set_possible_steps(filter_by_tasks=filter_by_tasks, tasks=tasks)
        if "similar_robot_obs" in goal_strategy_prob.keys():
            self.set_nn_steps_from_step(nn_steps_from_step_path, num_nn=num_nn)
        self.goal_sampling_prob = goal_sampling_prob

    def __len__(self):
        return len(self.possible_steps)

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

        logger.info(f"Building nn_steps_from_step for {data_type} ...")

        import faiss  # Only required to build nn_steps_from_step

        step_robot_obs = OrderedDict()
        for start_step, end_step in tqdm(self.ep_start_end_ids):
            for step in tqdm(range(start_step, end_step)):
                data = np.load(self.step_to_file[step], allow_pickle=True)
                step_robot_obs[step] = data["robot_obs"]

        robot_obs = np.stack(list(step_robot_obs.values())).astype("float32")
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
            logger.info(
                "Successfully saved nn_steps_from_step in "
                f"{str(nn_steps_from_step_path)}"
            )
        self.nn_steps_from_step = nn_steps_from_step[data_type]

    def increase_horizon(self, epoch):
        desired_horizon = self.initial_horizon + epoch * self.horizon_step
        self.current_horizon = min(desired_horizon, self.max_horizon)

    def increase_horizon_to(self, desired_horizon):
        self.current_horizon = min(desired_horizon, self.max_horizon)

    def set_lang_ann(self):
        lang_ann_file = self.data_dir / "lang_annotations/auto_lang_ann.npy"
        lang_ann_file = lang_ann_file.expanduser()
        assert lang_ann_file.is_file(), "Language annotation file not found"
        self.lang_ann = np.load(lang_ann_file, allow_pickle=True).item()

    def sample_goal_strategy(self):
        options = list(self.goal_strategy_prob.keys())
        prob = list(self.goal_strategy_prob.values())
        return np.random.choice(options, p=prob)

    def set_possible_steps(self, filter_by_tasks: bool = False, tasks: List[str] = []):
        """Create list of possible steps considering all steps except
        episode end steps"""
        if (self.data_dir / "split.json").is_file():
            with open(self.data_dir / "split.json") as f:
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
            self.ep_start_end_ids = np.load(self.data_dir / "ep_start_end_ids.npy")
            logger.info(
                f"Found 'ep_start_end_ids.npy' with {len(self.ep_start_end_ids)} "
                "episodes."
            )

        self.possible_steps = []
        for start_step, end_step in self.ep_start_end_ids:
            poss_steps = range(start_step, end_step)
            self.possible_steps.extend(list(poss_steps))
        self.possible_steps.sort()

        if filter_by_tasks:
            task_steps = []
            for i, task in enumerate(self.lang_ann["language"]["task"]):
                if task in tasks:
                    start_step, end_step = self.lang_ann["info"]["indx"][i]
                    task_steps.extend(list(range(start_step, end_step + 1)))
            self.possible_steps = list(set(self.possible_steps) & set(task_steps))

    def apply_transforms(self, data):
        if self.transform_manager is not None:
            transf_data = self.transform_manager(
                data, self.transf_type, device=self.device
            )
            return transf_data
        return data

    def get_file_from_step(self, step, transform=True):
        data = np.load(self.step_to_file[step], allow_pickle=True)
        filtered_data = {modality: data[modality] for modality in self.modalities}
        if transform:
            return self.apply_transforms(filtered_data)
        return filtered_data

    def find_episode_end(self, step):
        for start_step, end_step in self.ep_start_end_ids:
            if start_step <= step <= end_step:
                return end_step
        return None

    def find_task_end(self, step):
        for i, task in enumerate(self.lang_ann["language"]["task"]):
            start_step, end_step = self.lang_ann["info"]["indx"][i]
            if start_step <= step <= end_step:
                return end_step
        return None

    def get_random_future_step(self, start_step: int, end_step: int):
        """Obtains a random future step between start_step and
        end_step (inclusive)"""
        if start_step is None or start_step >= end_step + 1:
            return None
        step_options = np.arange(start_step, end_step + 1)
        return np.random.choice(step_options)

    def get_goal_step(self, step: int, strategy: str = "random"):
        if strategy == "random":
            step_options = self.possible_steps.copy()
            step_options.remove(step)
            goal_step = np.random.choice(step_options)
        elif strategy == "geometric":
            episode_end = self.find_episode_end(step)
            disp = np.random.default_rng().geometric(p=self.goal_sampling_prob)
            horizon_end_step = step + disp
            goal_step = min(episode_end, horizon_end_step)
        elif strategy == "increasing_horizon":
            episode_end_step = self.find_episode_end(step)
            horizon_end_step = step + self.current_horizon
            end_step = min(episode_end_step, horizon_end_step)
            goal_step = self.get_random_future_step(
                start_step=step + 1, end_step=end_step
            )
            if goal_step is None:
                return self.get_goal_step(step, strategy="random")
        elif strategy == "similar_robot_obs":
            step_options = self.nn_steps_from_step[step]
            if len(step_options) == 0:
                return self.get_goal_step(step, strategy="random")
            goal_step = np.random.choice(step_options)
        elif strategy == "next_state":
            goal_step = step + 1
        elif strategy == "episode_future":
            episode_end_step = self.find_episode_end(step)
            goal_step = self.get_random_future_step(
                start_step=step + 1, end_step=episode_end_step
            )
            if goal_step is None:
                return self.get_goal_step(step, strategy="random")
        elif strategy == "task_future":
            task_end_step = self.find_episode_end(step)
            goal_step = self.get_random_future_step(
                start_step=step + 1, end_step=task_end_step
            )
            if goal_step is None:
                return self.get_goal_step(step, strategy="episode_future")
        return goal_step

    def set_step_to_file(self):
        """Create mapping from step to file"""
        step_to_file = {}
        file_list = get_file_list(self.data_dir, extension=".npz")
        for file in file_list:
            step = int(file.stem.split("_")[-1])
            step_to_file[step] = file
        self.step_to_file = step_to_file

    def get_transition(self, step):
        obs = self.get_file_from_step(step, transform=True)
        action = obs.pop(self.action_type, None)
        next_obs = self.get_file_from_step(step + 1, transform=True)
        next_obs.pop(self.action_type, None)
        strategy = self.sample_goal_strategy()
        goal_step = self.get_goal_step(step=step, strategy=strategy)
        goal = self.get_file_from_step(goal_step, transform=True)
        reward = int(goal_step == step + 1)
        done = int(goal_step == step + 1)

        state = {"observation": obs, "goal": goal}
        next_state = {"observation": next_obs, "goal": goal}
        item = {
            "observations": state,
            "actions": action,
            "next_observations": next_state,
            "rewards": reward,
            "terminals": done,
        }
        return item

    def __getitem__(self, idx):
        step = self.possible_steps[idx]
        return self.get_transition(step)
