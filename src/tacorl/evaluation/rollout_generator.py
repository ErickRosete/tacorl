import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from random import shuffle

import numpy as np


class BaseRolloutGenerator(object):
    """
    Generates valid goal conditioned rollouts from the validation dataset
    """

    def __init__(
        self,
        data_dir: str = "~/thesis/calvin/validation",
        start_end_tasks: str = "~/thesis/calvin/start_end_tasks.json",
        strategy: str = "longest",
        min_seq_len: int = 16,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"{str(self.data_dir)} not found"
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

        start_end_tasks = Path(start_end_tasks).expanduser()
        assert start_end_tasks.is_file(), f"{str(start_end_tasks)} not found"
        with open(start_end_tasks) as f:
            start_end_tasks = json.load(f)
        self.rollout_tasks = self.build_rollout_tasks(start_end_tasks)
        self.order_rollouts(strategy)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_file_name(self, frame_idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_pattern[0]}"
            f"{frame_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def get_state_from_step(self, step: int, modalities=["rgb_static"]):
        file = self.get_file_name(step)
        state = np.load(file, allow_pickle=True)
        return {modality: state[modality] for modality in modalities}

    def get_state_info_from_step(self, step: int):
        file = self.get_file_name(step)
        state = np.load(file, allow_pickle=True)
        return {
            "robot_obs": state["robot_obs"],
            "scene_obs": state["scene_obs"],
        }

    def build_rollout_tasks(self, start_end_tasks):
        pass

    def order_rollouts(self, strategy):
        pass

    def get_rollout_tasks(self):
        return self.rollout_tasks


class SingleTaskRolloutGenerator(BaseRolloutGenerator):
    def build_rollout_tasks(self, start_end_tasks):
        rollout_tasks = {}
        for start_idx, end_tasks in start_end_tasks.items():
            for end_idx, completed_tasks in end_tasks.items():
                if len(completed_tasks) == 1:
                    task = completed_tasks[0]
                    if task not in rollout_tasks:
                        rollout_tasks[task] = []
                    start_step = int(start_idx)
                    end_step = int(end_idx)
                    seq_len = end_step - start_step
                    if self.max_seq_len > seq_len > self.min_seq_len:
                        rollout_tasks[task].append(
                            {
                                "start_step": start_step,
                                "end_step": end_step,
                                "seq_len": seq_len,
                            }
                        )
        return rollout_tasks

    def order_rollouts(self, strategy):
        if strategy == "shortest":
            for key, value in self.rollout_tasks.items():
                self.rollout_tasks[key] = sorted(value, key=lambda d: d["seq_len"])
        elif strategy == "longest":
            for key, value in self.rollout_tasks.items():
                self.rollout_tasks[key] = sorted(
                    value, key=lambda d: d["seq_len"], reverse=True
                )
        elif strategy == "random":
            for key in self.rollout_tasks.keys():
                shuffle(self.rollout_tasks[key])

    def get_reset_info(self, task, task_idx):
        rollout_task = self.rollout_tasks[task][task_idx]
        reset_info = {
            "task_info": {
                "start_info": self.get_state_info_from_step(rollout_task["start_step"]),
                "goal_info": self.get_state_info_from_step(rollout_task["end_step"]),
                "tasks": [task],
            }
        }
        return reset_info

    def get_rollout_task(self, task, task_idx):
        return self.rollout_tasks[task][task_idx]

    def get_num_rollouts_from_task(self, task):
        return len(self.rollout_tasks[task])


class LongHorizonRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, tasks_per_rollout: int = 4, *args, **kwargs):
        self.tasks_per_rollout = tasks_per_rollout
        super().__init__(*args, **kwargs)

    def build_rollout_tasks(
        self,
        start_end_tasks,
    ):
        rollout_tasks = []
        for start_idx, end_tasks in start_end_tasks.items():
            for end_idx, completed_tasks in end_tasks.items():
                if len(completed_tasks) == self.tasks_per_rollout:
                    rel_task = {
                        "start_step": int(start_idx),
                        "end_step": int(end_idx),
                        "seq_len": int(end_idx) - int(start_idx),
                        "completed_tasks": completed_tasks,
                    }
                    rollout_tasks.append(rel_task)
        return rollout_tasks

    def order_rollouts(self, strategy):
        if strategy == "shortest":
            self.rollout_tasks = sorted(self.rollout_tasks, key=lambda d: d["seq_len"])
        elif strategy == "longest":
            self.rollout_tasks = sorted(
                self.rollout_tasks, key=lambda d: d["seq_len"], reverse=True
            )
        elif strategy == "random":
            shuffle(self.rollout_tasks)

    def get_reset_info(self, task_idx):
        rollout_task = self.rollout_tasks[task_idx]
        reset_info = {
            "task_info": {
                "start_info": self.get_state_info_from_step(rollout_task["start_step"]),
                "goal_info": self.get_state_info_from_step(rollout_task["end_step"]),
                "tasks": rollout_task["completed_tasks"],
            }
        }
        return reset_info


class LongHorizonSequentialRolloutGenerator(BaseRolloutGenerator):
    def __init__(self, tasks_per_rollout: int = 5, *args, **kwargs):
        self.tasks_per_rollout = tasks_per_rollout
        super().__init__(*args, **kwargs)

    def build_rollout_tasks(
        self,
        start_end_tasks,
    ):

        filtered_start_end_tasks = OrderedDict()
        for start_idx, end_tasks in start_end_tasks.items():
            sorted_end_idcs = sorted(list([int(key) for key in end_tasks.keys()]))
            sorted_end_idcs = sorted_end_idcs[: self.tasks_per_rollout]
            task_counter = 1
            new_start_end_tasks_entry = OrderedDict()
            for end_idx in sorted_end_idcs:
                completed_tasks = end_tasks[str(end_idx)]
                # To test sequential tasks the number of completed tasks
                # must increase through time.
                if len(completed_tasks) != task_counter:
                    break
                new_start_end_tasks_entry[end_idx] = completed_tasks
                task_counter += 1

                if len(completed_tasks) == self.tasks_per_rollout:
                    filtered_start_end_tasks[start_idx] = new_start_end_tasks_entry
                    break

        return filtered_start_end_tasks

    def order_rollouts(self, strategy):
        if strategy == "shortest":
            self.rollout_tasks = OrderedDict(
                sorted(
                    self.rollout_tasks.items(),
                    key=lambda it: next(reversed(it[1])) - int(it[0]),
                )
            )
        elif strategy == "longest":
            self.rollout_tasks = OrderedDict(
                sorted(
                    self.rollout_tasks.items(),
                    key=lambda it: next(reversed(it[1])) - int(it[0]),
                    reverse=True,
                )
            )
        elif strategy == "random":
            rand_rollout_tasks = self.rollout_tasks.items()
            shuffle(rand_rollout_tasks)
            self.rollout_tasks = OrderedDict(rand_rollout_tasks)

    def get_reset_info(self, task_idx):
        rollout_task = self.rollout_tasks[task_idx]
        reset_info = {
            "task_info": {
                "start_info": self.get_state_info_from_step(rollout_task["start_step"]),
                "goal_info": self.get_state_info_from_step(rollout_task["end_step"]),
                "tasks": rollout_task["completed_tasks"],
            }
        }
        return reset_info
