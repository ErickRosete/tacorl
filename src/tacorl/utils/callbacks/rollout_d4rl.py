import math
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


class Rollout(Callback):
    """
    A class for performing rollouts.
    """

    def __init__(
        self,
        rollout_manager: DictConfig,
        val_episodes: int = 32,
        skip_first_n_epochs: int = 0,
        val_every_n_epochs: int = None,
        val_every_n_episodes: int = None,
        val_every_n_batches: int = None,
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
        self.skip_first_n_epochs = skip_first_n_epochs
        self.rollout_manager = hydra.utils.instantiate(rollout_manager)

    @torch.no_grad()
    def evaluate(
        self,
        pl_module: pl.LightningModule,
        num_episodes: int = 5,
        render: bool = False,
    ):
        log_rank_0("Evaluation episodes in process")
        pl_module.eval()

        val_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "score": [],
            "succesful_episodes": 0,
        }

        episodes = list(range(num_episodes))
        # Distribute rollouts between ranks
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            dist_num_episodes = world_size * math.ceil(num_episodes / world_size)
            episodes = [e for e in range(dist_num_episodes) if e % world_size == rank]

        for i, _ in tqdm(enumerate(episodes)):
            info = self.rollout_manager.episode_rollout(
                pl_module=pl_module,
                env=self.env,
                render=render,
            )
            val_info["episode_returns"].append(info["episode_return"])
            val_info["episodes_lengths"].append(info["episode_length"])
            val_info["score"].append(info["score"])
            val_info["succesful_episodes"] += int(info["success"])

        val_info = {
            "accuracy": val_info["succesful_episodes"] / len(episodes),
            "avg_episode_return": np.mean(val_info["episode_returns"]),
            "avg_episode_length": np.mean(val_info["succesful_episodes"]),
            "score": np.mean(val_info["score"]),
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

        val_info = self.evaluate(num_episodes=self.val_episodes, pl_module=pl_module)

        if "accuracy" in val_info and "avg_episode_return" in val_info:
            log_rank_0(
                "Validation | Accuracy: %.2f, Return: %.2f"
                % (val_info["accuracy"], val_info["avg_episode_return"])
            )

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

        return super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, unused=unused
        )

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
        pl_module.log("val_score", val_info["score"], on_epoch=True)
        return super().on_train_epoch_end(trainer, pl_module)
