import logging
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tacorl.datamodule.dataset.rl_dataset import RLDataset

logger = logging.getLogger(__name__)


class OnlineRLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        module: pl.LightningModule,
        batch_size: int = 32,
    ):
        super().__init__()
        self.module = module
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.train_dataset = RLDataset(
            replay_buffer=self.module.replay_buffer,
            batch_size=self.batch_size,
            transform_manager=self.module.transform_manager,
            device=self.module.device,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )
