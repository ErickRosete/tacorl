import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)


class IncreaseHorizonUncertainty(pl.Callback):
    def __init__(self, forward_passes: int = 3, std_threshold: float = 0.125) -> None:
        super().__init__()
        self.forward_passes = forward_passes
        self.std_threshold = std_threshold
        self.predictions = []

    @staticmethod
    def get_dataset(trainer: pl.Trainer):
        train_ds = trainer.datamodule.train_dataset
        if hasattr(train_ds, "dataset"):
            train_ds = train_ds.dataset
        return train_ds

    @staticmethod
    def enable_dropout(model):
        """Function to enable the dropout layers during test-time"""
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        pl_module.eval()
        critic = pl_module.actor_critic.critic
        self.enable_dropout(critic)
        (batch_observations, batch_actions, _, _, _) = pl_module.overwrite_batch(batch)
        batch_predictions = []
        for i in range(self.forward_passes):
            q1_pred, q2_pred = critic(batch_observations, batch_actions)
            batch_predictions.append(q1_pred)
            batch_predictions.append(q2_pred)
        batch_predictions = torch.stack(batch_predictions, dim=0)  # 2f, b, 1
        self.predictions.append(batch_predictions)
        pl_module.train()
        return super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, unused
        )

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        train_ds = self.get_dataset(trainer)
        if hasattr(train_ds, "goal_strategy_prob"):
            if "increasing_horizon" in train_ds.goal_strategy_prob.keys():
                predictions = torch.cat(self.predictions, dim=1)
                self.predictions = []
                if dist.is_available() and dist.is_initialized():
                    predictions = pl_module.all_gather(predictions)
                    predictions = predictions.permute(1, 0, 2, 3).contiguous()
                    predictions = predictions.view(predictions.shape[0], -1)
                    if dist.get_rank() != 0:
                        return
                std = torch.std(predictions, dim=0)
                avg_std = torch.mean(std)
                pl_module.log(
                    "train/goal_horizon",
                    torch.tensor(train_ds.current_horizon, dtype=torch.float),
                    on_epoch=True,
                    rank_zero_only=True,
                )
                pl_module.log(
                    "train/Q_avg_std", avg_std, on_epoch=True, rank_zero_only=True
                )
                if avg_std < self.std_threshold:
                    desired_horizon = train_ds.current_horizon + train_ds.horizon_step
                    train_ds.increase_horizon_to(desired_horizon)
        return super().on_epoch_end(trainer=trainer, pl_module=pl_module)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> dict:
        train_ds = self.get_dataset(trainer)
        if (
            hasattr(train_ds, "goal_strategy_prob")
            and "increasing_horizon" in train_ds.goal_strategy_prob.keys()
        ):
            checkpoint["current_horizon"] = train_ds.current_horizon
        return checkpoint

    def on_load_checkpoint(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callback_state: Dict[str, Any],
    ) -> None:
        train_ds = self.get_dataset(trainer)
        if (
            hasattr(train_ds, "goal_strategy_prob")
            and "increasing_horizon" in train_ds.goal_strategy_prob.keys()
        ):
            current_horizon = callback_state.get("current_horizon", None)
            if current_horizon is not None:
                train_ds.increase_horizon_to(current_horizon)
