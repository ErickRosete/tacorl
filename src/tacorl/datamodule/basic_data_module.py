import logging
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import tacorl
from tacorl.utils.episode_utils import load_dataset_statistics

logger = logging.getLogger(__name__)


class BasicDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        transform_manager: DictConfig = {},
        dataset: DictConfig = {},
        num_workers: int = 4,
        batch_size: int = 32,
        train_percentage: float = 1.0,
        val_percentage: float = 1.0,
        pin_memory: bool = True,
        shuffle_val: bool = False,
        create_vis_dataset: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_val = shuffle_val
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.create_vis_dataset = create_vis_dataset

        self.data_path = Path(data_dir).expanduser()
        if not self.data_path.is_absolute():
            self.data_path = Path(tacorl.__file__).parent / self.data_path

        self.split_by_file = False
        if (self.data_path / "training").is_dir():
            self.training_dir = self.data_path / "training"
            self.val_dir = self.data_path / "validation"
        elif (self.data_path / "split.json").is_file():
            self.split_by_file = True
        else:
            raise Exception(
                f"No training/validation partition found inside {str(self.data_path)}"
            )

        self.transform_manager_cfg = OmegaConf.to_container(
            transform_manager, resolve=True
        )
        self.dataset_cfg = OmegaConf.to_container(dataset, resolve=True)

    def prepare_data(self, *args, **kwargs):
        if self.split_by_file:
            check_dir = self.data_path
        else:
            check_dir = self.training_dir

        # Check if training dir contain npz files
        dataset_exist = len(list(check_dir.rglob("*.npz"))) > 0

        # Download and unpack images
        if not dataset_exist:
            logger.error(
                "Please download the dataset before starting a training!"
                "Specify the dataset path with "
                "datamodule.data_dir=/path/to/dataset/ "
            )
            exit(-1)

    def setup(self, stage: Optional[str] = None):

        train_dataset_cfg = self.dataset_cfg.copy()
        if self.val_percentage > 0:
            val_dataset_cfg = self.dataset_cfg.copy()
        if self.create_vis_dataset:
            vis_dataset_cfg = self.dataset_cfg.copy()

        # Instantiate transform manager
        train_dir = self.data_path if self.split_by_file else self.training_dir
        if len(self.transform_manager_cfg) > 0:
            transforms = load_dataset_statistics(
                train_dir, self.transform_manager_cfg["transforms"]
            )
            self.transform_manager_cfg["transforms"] = transforms
            self.transform_manager = hydra.utils.instantiate(self.transform_manager_cfg)
            # Train dataset
            train_dataset_cfg["transform_manager"] = self.transform_manager
            train_dataset_cfg["transf_type"] = "train"
            # Val dataset
            if self.val_percentage > 0:
                val_dataset_cfg["transform_manager"] = self.transform_manager
                val_dataset_cfg["transf_type"] = "validation"
            # Vis dataset
            if self.create_vis_dataset:
                vis_dataset_cfg["transform_manager"] = self.transform_manager
                vis_dataset_cfg["transf_type"] = "visualization"

        # Instantiate train_dataset
        self.train_dataset = hydra.utils.instantiate(
            train_dataset_cfg, data_dir=train_dir, train=True
        )
        train_dataset_len = self.train_dataset.__len__()
        new_indices = list(range(int(train_dataset_len * self.train_percentage)))
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, new_indices)

        # Instantiate val_dataset
        if self.val_percentage > 0:
            val_dir = self.data_path if self.split_by_file else self.val_dir
            self.val_dataset = hydra.utils.instantiate(
                val_dataset_cfg, data_dir=val_dir, train=False
            )
            val_dataset_len = self.val_dataset.__len__()
            new_indices = list(range(int(val_dataset_len * self.val_percentage)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, new_indices)

        # Vis dataset
        if self.create_vis_dataset:
            self.vis_dataset = hydra.utils.instantiate(
                vis_dataset_cfg, data_dir=val_dir, train=False
            )
            self.vis_dataset = torch.utils.data.Subset(self.vis_dataset, new_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_percentage > 0:
            return DataLoader(
                self.val_dataset,
                shuffle=self.shuffle_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def vis_dataloader(self) -> DataLoader:
        return DataLoader(
            self.vis_dataset,
            shuffle=self.shuffle_val,
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
