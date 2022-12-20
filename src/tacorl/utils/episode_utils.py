import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_task_info_of_sequence(env, initial_state_info, last_state_info) -> List:
    """
    This method checks which tasks where successfully performed in a sequence
    by resetting env to first and last state of the sequence.

    Output:
        task_info
    """

    tasks = env.tasks
    env.reset(**last_state_info)
    goal_info = env.get_info()
    # reset env to state of first step in the episode
    env.reset(**initial_state_info)
    start_info = env.get_info()

    # check if task was achieved in sequence
    task_info = tasks.get_task_info(start_info, goal_info)
    return task_info


def get_state_info_on_idx(
    state_info: Dict[str, Any] = None, batch_idx: int = 0, seq_idx: int = 0
):
    return {
        "robot_obs": state_info["robot_obs"][batch_idx, seq_idx],
        "scene_obs": state_info["scene_obs"][batch_idx, seq_idx],
    }


def get_state_info_dict(
    episode: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    :param episode: episode loaded by dataset loader
    :return: info dict of full robot and scene state (for env resets)
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }


def load_dataset_statistics(train_dataset_dir, transforms):
    """
    Tries to load statistics.yaml in train dataset folder in order to
    update the transforms hardcoded in the hydra config file.
    If no statistics.yaml exists, nothing is changed

    Args:
        train_dataset_dir: path of the training folder
        transforms: transforms loaded from hydra conf
    Returns:
        transforms: potentially updated transforms
    """
    statistics_path = Path(train_dataset_dir) / "statistics.yaml"
    if not statistics_path.is_file():
        return transforms

    statistics = OmegaConf.load(statistics_path)
    for transf_key in ["train", "validation"]:
        for modality in transforms[transf_key]:
            if modality in statistics:
                conf_transforms = transforms[transf_key][modality]
                dataset_transforms = statistics[modality]
                for dataset_trans in dataset_transforms:
                    # Use transforms from tacorl not calvin_agent
                    dataset_trans["_target_"] = dataset_trans["_target_"].replace(
                        "calvin_agent", "tacorl"
                    )
                    exists = False
                    for i, conf_trans in enumerate(conf_transforms):
                        if dataset_trans["_target_"] == conf_trans["_target_"]:
                            exists = True
                            transforms[transf_key][modality][i] = dataset_trans
                            break
                    if not exists:
                        transforms[transf_key][modality] = ListConfig(
                            [*conf_transforms, dataset_trans]
                        )
    return transforms
