import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="train")
def main(cfg):
    # Do not show tacto renderer output
    logger = logging.getLogger("tacto.renderer")
    logger.propagate = False
    logger = logging.getLogger(__name__)
    # Assert params
    if cfg.trainer.strategy == "ddp":
        if "num_parallel_envs" in cfg.module.keys():
            assert (
                cfg.module.num_parallel_envs <= 1
            ), "OOM issues may arise with DDP and several parallel envs"

        if "replay_buffer_path" in cfg.module.keys():
            assert (
                cfg.module.replay_buffer_path is None
            ), "Saving replay buffer path is not possible with ddp"

    logger.info("Initializing callbacks")
    callbacks = [
        hydra.utils.instantiate(call_def) for call_def in cfg.callbacks.values()
    ]
    pl_logger = hydra.utils.instantiate(cfg.logger)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["callbacks"] = callbacks
    trainer_cfg["logger"] = pl_logger
    trainer = pl.Trainer(**trainer_cfg)

    logger.info("Initializing pl module")
    model = hydra.utils.instantiate(cfg.module)

    # Add model to datamodule for online rl
    logger.info("Initializing datamodule")
    datamodule_cfg = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if datamodule_cfg["_target_"].split(".")[-1] == "OnlineRLDataModule":
        datamodule_cfg["module"] = model
    datamodule = hydra.utils.instantiate(datamodule_cfg)

    # Load last checkpoint before starting training
    ckpt_path = None
    if "checkpoint" in cfg.callbacks.keys():
        model_dir = (
            Path(cfg.callbacks.checkpoint.dirpath)
            if cfg.callbacks.checkpoint.dirpath is not None
            else None
        )
        if model_dir is not None:
            ckpt_path = model_dir / "last.ckpt"
            if ckpt_path.is_file():
                ckpt_path = ckpt_path
                logger.info(f"Checkpoint {str(ckpt_path)} found")
                logger.info("Training will be resumed from checkpoint")
            else:
                ckpt_path = None

    if ckpt_path is None:
        logger.info("Checkpoint NOT found, starting train from scratch")
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=datamodule)

    if hasattr(model, "env"):
        model.env.close()


if __name__ == "__main__":
    import os

    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    main()
