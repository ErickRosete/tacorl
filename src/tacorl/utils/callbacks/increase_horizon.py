import pytorch_lightning as pl
import torch


class IncreaseHorizonLinear(pl.Callback):
    @staticmethod
    def get_dataset(trainer: pl.Trainer):
        train_ds = trainer.datamodule.train_dataset
        if hasattr(train_ds, "dataset"):
            train_ds = train_ds.dataset
        return train_ds

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        train_ds = self.get_dataset(trainer)
        if hasattr(train_ds, "goal_strategy_prob"):
            pl_module.log(
                "train/goal_horizon",
                torch.tensor(train_ds.current_horizon, dtype=torch.float),
                on_epoch=True,
            )
            if "increasing_horizon" in train_ds.goal_strategy_prob.keys():
                train_ds.increase_horizon(epoch=pl_module.current_epoch + 1)
        return super().on_epoch_end(trainer=trainer, pl_module=pl_module)


class IncreaseHorizonConstant(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        return super().on_epoch_end(trainer=trainer, pl_module=pl_module)
