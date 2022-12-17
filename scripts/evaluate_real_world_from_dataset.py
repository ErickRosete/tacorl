import json
import logging
import os
import re
from pathlib import Path
from typing import List

import cv2
import gym
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from robot_io.cams.realsense.realsense import Realsense  # noqa

from tacorl.utils.networks import (
    load_pl_module_from_checkpoint,
    load_transform_manager_from_dir,
)
from tacorl.utils.transforms import TransformManager

logger = logging.getLogger(__name__)


def upscale_img(img, max_width: int = 500):
    res = img.shape[:2][::-1]
    scale = max_width / max(res)
    new_res = tuple((np.array(res) * scale).astype(int))
    return cv2.resize(img, new_res)


def show_img(title: str, img: np.ndarray, wait, resize: bool = True):
    if resize:
        cv2.imshow(title, upscale_img(img[:, :, ::-1]))
    else:
        cv2.imshow(title, img[:, :, ::-1])
    return cv2.waitKey(wait) % 256


class StartGoalProposer:
    def __init__(
        self,
        data_dir: str,
        window_size: int = 128,
        train: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.train = train
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir()
        self.window_size = window_size
        self.set_episode_lookup()
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

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

    def set_ep_start_end_ids(self):
        """Extracts ep_start_end_ids from the adequate partition stored in split.json,
        if it doesn't have this file it searches for ep_start_end_ids.npy"""

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

    def get_file_name(self, frame_idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_pattern[0]}"
            f"{frame_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def set_episode_lookup(self):
        """Sets possible sequences that can be used as goals"""
        self.set_ep_start_end_ids()

        episode_lookup = []
        for start_idx, end_idx in self.ep_start_end_ids:
            assert end_idx > self.window_size
            for idx in range(start_idx, end_idx + 1 - self.window_size):
                episode_lookup.append(idx)
        self.episode_lookup = episode_lookup

    def __len__(self):
        return len(self.episode_lookup)

    def propose_goal(self, start_idx):
        start_step = self.episode_lookup[start_idx]
        end_step = start_step + self.window_size - 1
        start_state = np.load(self.get_file_name(start_step), allow_pickle=True)
        end_state = np.load(self.get_file_name(end_step), allow_pickle=True)
        return start_state, end_state


class EvaluationManager:
    def __init__(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        transform_manager: TransformManager,
        start_goal_proposer: DictConfig = None,
        rollout_manager: DictConfig = None,
        render: bool = True,
        reset_from_robot_obs: bool = True,
        reset_to_neutral: bool = False,
        goal_modalities: List[str] = ["rgb_static", "rgb_gripper"],
    ):
        self.pl_module = pl_module
        self.env = env
        self.start_goal_proposer = StartGoalProposer(**start_goal_proposer)
        self.transform_manager = transform_manager
        print("rollout_manager: ", rollout_manager)
        self.rollout_manager = hydra.utils.instantiate(
            rollout_manager, transform_manager=self.transform_manager
        )
        self.num_seq = len(self.start_goal_proposer)
        self.render = render
        self.goal_modalities = goal_modalities
        self.reset_from_robot_obs = reset_from_robot_obs
        self.reset_to_neutral = reset_to_neutral

    def evaluate_policy_dataset(self):
        i = 0
        logger.info("Press A / D to move through episodes, E / Q to skip 50 episodes.")
        logger.info(
            "Press O to run inference with the model, but use goal from episode."
        )

        while 1:
            start_state, end_state = self.start_goal_proposer.propose_goal(i)
            show_img("start", start_state["rgb_static"], wait=1, resize=True)
            show_img("start_gripper", start_state["rgb_gripper"], wait=1, resize=True)
            show_img("goal_gripper", end_state["rgb_gripper"], wait=1, resize=True)
            k = show_img("goal", end_state["rgb_static"], wait=0, resize=True)

            if k == ord("a"):
                i -= 1
                i = int(np.clip(i, 0, self.num_seq))
            if k == ord("d"):
                i += 1
                i = int(np.clip(i, 0, self.num_seq))
            if k == ord("q"):
                i -= 50
                i = int(np.clip(i, 0, self.num_seq))
            if k == ord("e"):
                i += 50
                i = int(np.clip(i, 0, self.num_seq))
            if k == ord("o"):
                goal = {}
                for modality in self.goal_modalities:
                    goal[modality] = end_state[modality]
                reset_info = {"goal": goal}
                reset_info["reset_to_neutral"] = self.reset_to_neutral
                if self.reset_from_robot_obs:
                    reset_info["robot_obs"] = start_state["robot_obs"]
                self.rollout(reset_info)

    @torch.no_grad()
    def rollout(self, reset_info):
        logger.info("Starting evaluation rollout ...")
        self.pl_module.eval()
        self.rollout_manager.episode_rollout(
            pl_module=self.pl_module,
            env=self.env,
            reset_info=reset_info,
            render=self.render,
        )
        self.pl_module.train()
        logger.info("Finished evaluation rollout")

    def evaluate_policy_sequences(self):
        eval_id = {}
        eval_id["push_red_button"] = 205
        eval_id["push_blue_button"] = 87
        eval_id["push_green_button"] = 442
        eval_id["open_drawer"] = 723
        eval_id["close_drawer"] = 841
        eval_id["open_drawer_diffent"] = 998
        eval_id["close_drawer_diffent"] = 1122
        eval_id["move_sliding_door_middle"] = 1272
        eval_id["move_sliding_door_left"] = 1293
        eval_id["move_sliding_door_right"] = 1435
        eval_id["pink_block_inside_container"] = 2385
        eval_id["pink_block_top_left"] = 1874
        eval_id["pink_block_middle"] = 2029
        eval_id["pink_block_on_top_drawer"] = 2141
        eval_id["lift_pink_block"] = 2317
        eval_id["place_pink_block_top_right"] = 3124
        eval_id["stack_pink_on_purple"] = 3252
        eval_id["rotate_pink_left"] = 3451
        eval_id["rotate_pink_right"] = 3590
        eval_id["slide_pink_left_centered"] = 3720
        eval_id["slide_pink_right_centered"] = 3825
        eval_id["slide_pink_left_extreme"] = 3900
        eval_seq = [
            ["push_green_button", "lift_pink_block"],
            ["push_green_button", "push_blue_button", "push_red_button"],
            ["push_green_button", "open_drawer", "move_sliding_door_left"],
            ["lift_pink_block", "place_pink_block_top_right"],
            ["lift_pink_block", "pink_block_inside_container"],
            ["lift_pink_block", "pink_block_top_left"],
            ["lift_pink_block", "pink_block_on_top_drawer"],
            ["rotate_pink_left", "pink_block_on_top_drawer"],
            ["rotate_pink_right", "pink_block_inside_container"],
        ]
        logger.info("Press A / D to move through eval sequences.")
        logger.info(
            "Press O to run inference with the model, but use goal from episode."
        )
        i = 0
        self.env._max_episode_steps = 100
        printed_flag = False
        while 1:
            act_seq = eval_seq[i]
            if not printed_flag:
                logger.info(act_seq)
                printed_flag = True
            # list_subtask_ids =[]
            for idx, subtask in enumerate(act_seq):
                curr_goal = eval_id[subtask]
                start_state, end_state = self.start_goal_proposer.propose_goal(
                    curr_goal
                )
                show_img(
                    "goal_gripper_" + str(idx),
                    end_state["rgb_gripper"],
                    wait=1,
                    resize=True,
                )
                show_img(
                    "goal_" + str(idx), end_state["rgb_static"], wait=1, resize=True
                )
            k = show_img(
                "goal_" + str(idx),
                self.start_goal_proposer.propose_goal(eval_id[act_seq[-1]])[1][
                    "rgb_static"
                ],
                wait=0,
                resize=True,
            )
            if k == ord("a"):
                i -= 1
                i = int(np.clip(i, 0, self.num_seq))
                printed_flag = False
                cv2.destroyAllWindows()
            if k == ord("d"):
                i += 1
                i = int(np.clip(i, 0, self.num_seq))
                printed_flag = False
                cv2.destroyAllWindows()

            if k == ord("o"):
                printed_flag = False
                for subtask in act_seq:
                    goal = {}
                    start_state, end_state = self.start_goal_proposer.propose_goal(
                        (eval_id[subtask])
                    )
                    for modality in self.goal_modalities:
                        goal[modality] = end_state[modality]
                    reset_info = {"goal": goal}
                    reset_info["reset_to_neutral"] = self.reset_to_neutral
                    if self.reset_from_robot_obs:
                        reset_info["robot_obs"] = start_state["robot_obs"]
                    logger.info("rolling out %s", subtask)
                    self.rollout(reset_info)

    def __call__(self):
        # self.evaluate_policy_dataset()
        self.evaluate_policy_sequences()


@hydra.main(config_path="../config", config_name="evaluate_real_world_from_dataset")
def main(cfg):
    # Init module
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(module_path, epoch=epoch).cuda()
    transform_manager = load_transform_manager_from_dir(module_path)
    modalities = list(pl_module.all_modalities)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot, modalities=modalities)

    # Start eval
    eval_manager = EvaluationManager(
        pl_module=pl_module,
        env=env,
        transform_manager=transform_manager,
        **cfg.evaluation,
    )
    eval_manager()


if __name__ == "__main__":
    main()
