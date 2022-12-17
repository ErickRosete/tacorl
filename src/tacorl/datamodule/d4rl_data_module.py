import logging
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class D4RLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform_manager: DictConfig = {},
        dataset: DictConfig = {},
        num_workers: int = 4,
        batch_size: int = 32,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform_manager_cfg = OmegaConf.to_container(
            transform_manager, resolve=True
        )
        self.dataset_cfg = OmegaConf.to_container(dataset, resolve=True)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        train_dataset_cfg = self.dataset_cfg.copy()
        # Instantiate transform manager
        if len(self.transform_manager_cfg) > 0:
            self.transform_manager = hydra.utils.instantiate(self.transform_manager_cfg)
            # Train dataset
            train_dataset_cfg["transform_manager"] = self.transform_manager
            train_dataset_cfg["transf_type"] = "train"

        # Instantiate train_dataset
        self.train_dataset = hydra.utils.instantiate(train_dataset_cfg)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


@hydra.main(config_path="../../config", config_name="test/datamodule_test")
def main(cfg):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        print()


if __name__ == "__main__":
    main()
