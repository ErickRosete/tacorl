import hydra
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf


@hydra.main(config_path="../../../config", config_name="test/callback_test")
def main(cfg):

    callbacks = [
        hydra.utils.instantiate(call_def) for call_def in cfg.callbacks.values()
    ]
    logger = hydra.utils.instantiate(cfg.logger)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["callbacks"] = callbacks
    trainer_cfg["logger"] = logger
    trainer_cfg["val_check_interval"] = 1.0
    trainer = pl.Trainer(**trainer_cfg)
    model = hydra.utils.instantiate(cfg.module)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
