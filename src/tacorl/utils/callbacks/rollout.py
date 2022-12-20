import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from tacorl.utils.misc import log_rank_0, pl_log_metrics
from tacorl.utils.path import get_file_list
from tacorl.utils.wandb_loggers.video_logger import VideoLogger


class Rollout(Callback):
    """
    A class for performing rollouts.
    """

    def __init__(
        self,
        rollout_manager: DictConfig,
        val_episodes: int = 5,
        max_episode_steps: int = 100,
        skip_first_n_epochs: int = 0,
        val_every_n_epochs: int = None,
        val_every_n_episodes: int = None,
        val_every_n_batches: int = None,
        eval_strategy: str = "all_tasks",
        data_dir: str = "~/tacorl/calvin/validation",
        start_end_tasks: str = "~/tacorl/calvin/start_end_tasks.json",
        id_selection_strategy: str = "shortest",
        num_rollouts_per_task: int = 3,
        min_seq_len: int = 16,
        max_seq_len: int = 64,
    ):
        """
        If the env is goal conditioned and eval_all_tasks is set to true
        it will evaluate all list of possible goals, k rollouts
        per goal (num val_episodes will be ignored)
        Args:
            env: Environment where to perform rollouts, it can be different
                 than the training env
        """

        assert (
            val_every_n_epochs is not None
            or val_every_n_episodes is not None
            or val_every_n_batches is not None
        ), "val_every_n ... is not set"
        self.val_every_n_episodes = val_every_n_episodes
        self.val_every_n_epochs = val_every_n_epochs
        self.val_every_n_batches = val_every_n_batches
        self.val_episodes = val_episodes
        self.eval_strategy = eval_strategy
        self.skip_first_n_epochs = skip_first_n_epochs
        self.max_episode_steps = max_episode_steps
        self.rollout_manager = rollout_manager

        if self.eval_strategy == "all_tasks":
            self.num_rollouts_per_task = num_rollouts_per_task
            self.min_seq_len = min_seq_len
            self.max_seq_len = max_seq_len

            self.data_dir = Path(data_dir).expanduser()
            assert self.data_dir.is_dir(), f"{str(self.data_dir)} not found"

            self.set_step_to_file()

            start_end_tasks = Path(start_end_tasks).expanduser()
            assert start_end_tasks.is_file(), f"{str(start_end_tasks)} not found"
            with open(start_end_tasks) as f:
                start_end_tasks = json.load(f)
            self.rollout_tasks = self.get_rollout_tasks(
                start_end_tasks, id_selection_strategy
            )

    def set_step_to_file(self):
        """Create mapping from step to file"""
        step_to_file = {}
        file_list = get_file_list(self.data_dir)
        for file in file_list:
            step = int(file.stem.split("_")[-1])
            step_to_file[step] = file
        self.step_to_file = step_to_file

    def get_state_info_from_step(self, step: int):
        state = np.load(self.step_to_file[step], allow_pickle=True)
        return {
            "robot_obs": state["robot_obs"],
            "scene_obs": state["scene_obs"],
        }

    def get_rollout_tasks(
        self,
        start_end_tasks,
        id_selection_strategy: str = "shortest",
    ):

        rollout_tasks = {}
        for start_idx, end_tasks in start_end_tasks.items():
            for end_idx, completed_tasks in end_tasks.items():
                if len(completed_tasks) == 1:
                    task = completed_tasks[0]
                    if task not in rollout_tasks:
                        rollout_tasks[task] = []
                    seq_len = int(end_idx) - int(start_idx)
                    if self.max_seq_len > seq_len > self.min_seq_len:
                        rollout_tasks[task].append(
                            {
                                "start_step": int(start_idx),
                                "end_step": int(end_idx),
                                "seq_len": seq_len,
                            }
                        )
        if id_selection_strategy == "shortest":
            for key, value in rollout_tasks.items():
                rollout_tasks[key] = sorted(value, key=lambda d: d["seq_len"])
        return rollout_tasks

    @torch.no_grad()
    def evaluate_all_tasks(
        self,
        pl_module: pl.LightningModule = None,
        render: bool = False,
        log_type: str = "validation",
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        log_rank_0("Evaluation episodes in process")
        pl_module.eval()

        val_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "succesful_episodes": 0,
            "total_episodes": 0,
        }
        static_val_info = deepcopy(val_info)
        dynamic_val_info = deepcopy(val_info)

        for task in self.rollout_tasks.keys():
            log_rank_0(f"Evaluating task: {task}")
            task_info = {
                "episode_returns": [],
                "episodes_lengths": [],
                "succesful_episodes": 0,
            }

            if len(self.rollout_tasks[task]) > 0:
                goal_list = list(range(self.num_rollouts_per_task))

                # Distribute rollouts between ranks
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    num_goals = world_size * math.ceil(
                        self.num_rollouts_per_task / world_size
                    )
                    goal_list = [g for g in range(num_goals) if g % world_size == rank]

                # Make sure goal list is valid
                goal_list = [g % len(self.rollout_tasks[task]) for g in goal_list]
                for i, task_index in tqdm(enumerate(goal_list)):
                    rollout_task = self.rollout_tasks[task][task_index]
                    reset_info = {
                        "task_info": {
                            "start_info": self.get_state_info_from_step(
                                rollout_task["start_step"]
                            ),
                            "goal_info": self.get_state_info_from_step(
                                rollout_task["end_step"]
                            ),
                            "tasks": [task],
                        }
                    }
                    info = self.rollout_manager.episode_rollout(
                        pl_module=pl_module,
                        env=self.env,
                        reset_info=reset_info,
                        render=render,
                        video_logger=self.video_logger,
                        log_video=(i == 0),
                        task=task,
                    )
                    task_info["episode_returns"].append(info["episode_return"])
                    task_info["episodes_lengths"].append(info["episode_length"])
                    task_info["succesful_episodes"] += int(info["success"])

                if "block" in task:
                    val_info = dynamic_val_info
                else:
                    val_info = static_val_info
                val_info["episode_returns"].extend(task_info["episode_returns"])
                val_info["episodes_lengths"].extend(task_info["episodes_lengths"])
                val_info["succesful_episodes"] += task_info["succesful_episodes"]
                val_info["total_episodes"] += len(goal_list)
                task_info = {
                    "accuracy": task_info["succesful_episodes"] / len(goal_list),
                    "avg_episode_return": np.mean(task_info["episode_returns"]),
                    "avg_episode_length": np.mean(task_info["episodes_lengths"]),
                }
                pl_log_metrics(
                    metrics=task_info,
                    pl_module=pl_module,
                    log_type=f"{log_type}/{task}",
                    on_epoch=on_epoch,
                    on_step=on_step,
                    sync_dist=True,
                )
        dynamic_val_info = {
            "accuracy": dynamic_val_info["succesful_episodes"]
            / dynamic_val_info["total_episodes"],
            "avg_episode_return": np.mean(dynamic_val_info["episode_returns"]),
            "avg_episode_length": np.mean(dynamic_val_info["episodes_lengths"]),
        }
        pl_log_metrics(
            metrics=dynamic_val_info,
            pl_module=pl_module,
            log_type=f"{log_type}/dynamic",
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=True,
        )
        static_val_info = {
            "accuracy": static_val_info["succesful_episodes"]
            / static_val_info["total_episodes"],
            "avg_episode_return": np.mean(static_val_info["episode_returns"]),
            "avg_episode_length": np.mean(static_val_info["episodes_lengths"]),
        }
        pl_log_metrics(
            metrics=static_val_info,
            pl_module=pl_module,
            log_type=f"{log_type}/static",
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=True,
        )
        val_info = {
            "accuracy": (dynamic_val_info["accuracy"] + static_val_info["accuracy"])
            / 2,
            "avg_episode_return": (
                dynamic_val_info["avg_episode_return"]
                + static_val_info["avg_episode_return"]
            )
            / 2,
            "avg_episode_length": (
                dynamic_val_info["avg_episode_length"]
                + static_val_info["avg_episode_length"]
            )
            / 2,
        }
        pl_module.train()
        return val_info

    @torch.no_grad()
    def evaluate_env_tasks(
        self,
        pl_module: pl.LightningModule = None,
        render: bool = False,
        log_type: str = "validation",
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        log_rank_0("Evaluation episodes in process")
        pl_module.eval()

        val_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "succesful_episodes": 0,
            "total_episodes": 0,
        }
        static_val_info = deepcopy(val_info)
        dynamic_val_info = deepcopy(val_info)

        possible_tasks = self.env.get_possible_tasks()
        for task, num_goals in possible_tasks.items():
            log_rank_0(f"Evaluating task: {task}")
            task_info = {
                "episode_returns": [],
                "episodes_lengths": [],
                "succesful_episodes": 0,
            }

            goal_list = list(range(num_goals))
            # Distribute rollouts between ranks
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                dist_num_goals = world_size * math.ceil(num_goals / world_size)
                goal_list = [
                    g % num_goals
                    for g in range(dist_num_goals)
                    if g % world_size == rank
                ]

            for i, task_index in tqdm(enumerate(goal_list)):
                reset_info = {"task_info": {"task": task, "index": task_index}}
                info = self.rollout_manager.episode_rollout(
                    pl_module=pl_module,
                    env=self.env,
                    reset_info=reset_info,
                    render=render,
                    video_logger=self.video_logger,
                    log_video=(i == 0),
                    task=task,
                )
                task_info["episode_returns"].append(info["episode_return"])
                task_info["episodes_lengths"].append(info["episode_length"])
                task_info["succesful_episodes"] += int(info["success"])

            if "block" in task:
                val_info = dynamic_val_info
            else:
                val_info = static_val_info

            val_info["episode_returns"].extend(task_info["episode_returns"])
            val_info["episodes_lengths"].extend(task_info["episodes_lengths"])
            val_info["succesful_episodes"] += task_info["succesful_episodes"]
            val_info["total_episodes"] += len(goal_list)

            task_info = {
                "accuracy": task_info["succesful_episodes"] / len(goal_list),
                "avg_episode_return": np.mean(task_info["episode_returns"]),
                "avg_episode_length": np.mean(task_info["episodes_lengths"]),
            }
            pl_log_metrics(
                metrics=task_info,
                pl_module=pl_module,
                log_type=f"{log_type}/{task}",
                on_epoch=on_epoch,
                on_step=on_step,
                sync_dist=True,
            )

        dynamic_val_info = {
            "accuracy": dynamic_val_info["succesful_episodes"]
            / dynamic_val_info["total_episodes"],
            "avg_episode_return": np.mean(dynamic_val_info["episode_returns"]),
            "avg_episode_length": np.mean(dynamic_val_info["episodes_lengths"]),
        }
        pl_log_metrics(
            metrics=dynamic_val_info,
            pl_module=pl_module,
            log_type=f"{log_type}/dynamic",
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=True,
        )
        static_val_info = {
            "accuracy": static_val_info["succesful_episodes"]
            / static_val_info["total_episodes"],
            "avg_episode_return": np.mean(static_val_info["episode_returns"]),
            "avg_episode_length": np.mean(static_val_info["episodes_lengths"]),
        }
        pl_log_metrics(
            metrics=static_val_info,
            pl_module=pl_module,
            log_type=f"{log_type}/static",
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=True,
        )
        val_info = {
            "accuracy": (dynamic_val_info["accuracy"] + static_val_info["accuracy"])
            / 2,
            "avg_episode_return": (
                dynamic_val_info["avg_episode_return"]
                + static_val_info["avg_episode_return"]
            )
            / 2,
            "avg_episode_length": (
                dynamic_val_info["avg_episode_length"]
                + static_val_info["avg_episode_length"]
            )
            / 2,
        }

        pl_module.train()
        return val_info

    @torch.no_grad()
    def evaluate(
        self,
        pl_module: pl.LightningModule,
        num_episodes: int = 5,
        render: bool = False,
        device: str = "cuda",
    ):
        log_rank_0("Evaluation episodes in process")
        pl_module.eval()

        val_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "succesful_episodes": 0,
        }

        episodes = list(range(num_episodes))
        # Distribute rollouts between ranks
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            dist_num_episodes = world_size * math.ceil(num_episodes / world_size)
            episodes = [e for e in range(dist_num_episodes) if e % world_size == rank]

        for i, episode in tqdm(enumerate(episodes)):
            info = self.rollout_manager.episode_rollout(
                pl_module=pl_module,
                env=self.env,
                render=render,
                video_logger=self.video_logger,
                log_video=(i == 0),
            )
            val_info["episode_returns"].append(info["episode_return"])
            val_info["episodes_lengths"].append(info["episode_length"])
            val_info["succesful_episodes"] += int(info["success"])

        val_info = {
            "accuracy": val_info["succesful_episodes"] / len(episodes),
            "avg_episode_return": np.mean(val_info["episode_returns"]),
            "avg_episode_length": np.mean(val_info["succesful_episodes"]),
        }

        pl_module.train()
        return val_info

    def run_and_log_validation(
        self,
        pl_module: pl.LightningModule,
        log_type: str = "validation",
        on_step: bool = True,
        on_epoch: bool = True,
    ):
        self.env._max_episode_steps = self.max_episode_steps

        if self.eval_strategy == "all_tasks":
            val_info = self.evaluate_all_tasks(
                pl_module=pl_module,
                log_type=log_type,
                on_step=on_step,
                on_epoch=on_epoch,
            )
        elif self.eval_strategy == "env_tasks" and hasattr(
            self.env, "get_possible_tasks"
        ):
            val_info = self.evaluate_env_tasks(
                pl_module=pl_module,
                log_type=log_type,
                on_step=on_step,
                on_epoch=on_epoch,
            )
        else:
            val_info = self.evaluate(
                num_episodes=self.val_episodes, pl_module=pl_module
            )

        if "accuracy" in val_info and "avg_episode_return" in val_info:
            log_rank_0(
                "Validation | Accuracy: %.2f, Return: %.2f"
                % (val_info["accuracy"], val_info["avg_episode_return"])
            )

        self.video_logger.log("%s/rollout" % log_type)

        pl_log_metrics(
            metrics=val_info,
            pl_module=pl_module,
            log_type=log_type,
            on_epoch=on_epoch,
            on_step=on_step,
            sync_dist=True,
        )

        return val_info

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Setup logging classes"""
        if not hasattr(self, "env"):
            self.env = pl_module.env
        if not hasattr(self, "video_logger"):
            self.video_logger = VideoLogger(pl_module.logger)
        if not hasattr(self, "transform_manager"):
            self.transform_manager = trainer.datamodule.transform_manager
            self.rollout_manager = hydra.utils.instantiate(
                self.rollout_manager, transform_manager=self.transform_manager
            )
        return super().on_fit_start(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:

        if (
            pl_module.current_epoch >= self.skip_first_n_epochs
            and self.val_every_n_batches is not None
            and batch_idx % self.val_every_n_batches == 0
        ):
            self.run_and_log_validation(pl_module, log_type="batch_val")

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run episode validation by simulating forward passes in the environment"""
        val_info = {"avg_episode_return": float("-inf"), "accuracy": float("-inf")}

        if pl_module.current_epoch >= self.skip_first_n_epochs:
            episode_cond = (
                self.val_every_n_episodes is not None
                and hasattr(pl_module, "episode_number")
                and pl_module.episode_number.item() % self.val_every_n_episodes == 0
                and hasattr(pl_module, "episode_done")
                and pl_module.episode_done
            )
            epoch_cond = (
                self.val_every_n_epochs is not None
                and pl_module.current_epoch % self.val_every_n_epochs == 0
            )
            if episode_cond or epoch_cond:
                val_info = self.run_and_log_validation(
                    pl_module, on_step=False, on_epoch=True
                )
                if hasattr(pl_module, "save_checkpoint"):
                    pl_module.save_checkpoint()

        # Monitored metric to save top k model
        pl_module.log(
            "val_episode_return", val_info["avg_episode_return"], on_epoch=True
        )
        pl_module.log("val_accuracy", val_info["accuracy"], on_epoch=True)
        return super().on_train_epoch_end(trainer, pl_module)
