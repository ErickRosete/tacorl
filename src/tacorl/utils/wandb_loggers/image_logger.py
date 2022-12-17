from typing import List

import numpy as np
from pytorch_lightning.loggers import WandbLogger


class ImageLogger:
    def __init__(self, logger: WandbLogger) -> None:
        self.logger = logger

    def log(
        self,
        images: List[np.ndarray],
        caption: List[str] = ["observation"],
        name: str = "validation/last_state",
    ):

        wandb_logger = self.logger
        wandb_logger.log_image(key=name, images=images, caption=caption)
