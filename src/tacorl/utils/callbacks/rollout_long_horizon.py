import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from tqdm import tqdm

from tacorl.utils.callbacks.rollout import Rollout
from tacorl.utils.misc import log_rank_0


class RolloutLongHorizon(Rollout):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        num_rollouts: int = 10,
        tasks_per_rollout: int = 4,
        *args,
        **kwargs,
    ):
        self.num_rollouts = num_rollouts
        self.tasks_per_rollout = tasks_per_rollout
        kwargs["eval_strategy"] = "all_tasks"
        super().__init__(*args, **kwargs)

    def get_rollout_tasks(
        self,
        start_end_tasks,
        id_selection_strategy: str = "shortest",
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
        if id_selection_strategy == "shortest":
            rollout_tasks = sorted(rollout_tasks, key=lambda d: d["seq_len"])
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
        if len(self.rollout_tasks) == 0:
            return

        log_rank_0("Long Horizon rollouts evaluation in process")
        pl_module.eval()

        val_info = {
            "episode_returns": [],
            "episodes_lengths": [],
        }
        successful_tasks_per_rollout = np.array([0] * self.tasks_per_rollout)
        rollout_list = list(range(self.num_rollouts))

        # Distribute rollouts between ranks
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            num_rollouts = world_size * math.ceil(self.num_rollouts / world_size)
            rollout_list = [r for r in range(num_rollouts) if r % world_size == rank]

        # Make sure rollout_list is valid
        rollout_list = [r % len(self.rollout_tasks) for r in rollout_list]
        for i, task_index in tqdm(enumerate(rollout_list)):
            rollout_task = self.rollout_tasks[task_index]
            reset_info = {
                "task_info": {
                    "start_info": self.get_state_info_from_step(
                        rollout_task["start_step"]
                    ),
                    "goal_info": self.get_state_info_from_step(
                        rollout_task["end_step"]
                    ),
                    "tasks": rollout_task["completed_tasks"],
                }
            }
            info = self.rollout_manager.episode_rollout(
                pl_module=pl_module,
                env=self.env,
                reset_info=reset_info,
                render=render,
                video_logger=self.video_logger,
                log_video=i < 3,
                task=f"long_horizon_{i+1}",
            )
            val_info["episode_returns"].append(info["episode_return"])
            val_info["episodes_lengths"].append(info["episode_length"])

            num_successful_tasks = len(info["successful_tasks"])
            if num_successful_tasks > 0:
                successful_tasks_per_rollout[:num_successful_tasks] += 1

        val_info = {
            "LH_avg_episode_return": np.mean(val_info["episode_returns"]),
            "LH_avg_episode_length": np.mean(val_info["episodes_lengths"]),
        }
        for i in range(self.tasks_per_rollout):
            val_info[f"LH_{i + 1}_accuracy"] = successful_tasks_per_rollout[i] / len(
                rollout_list
            )

        pl_module.train()
        return val_info

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run episode validation by simulating forward passes in the environment"""
        epoch_cond = (
            pl_module.current_epoch >= self.skip_first_n_epochs
            and self.val_every_n_epochs is not None
            and pl_module.current_epoch % self.val_every_n_epochs == 0
        )
        if epoch_cond:
            self.run_and_log_validation(pl_module, on_step=False, on_epoch=True)
