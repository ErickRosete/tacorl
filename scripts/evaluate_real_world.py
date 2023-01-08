from pathlib import Path

import cv2
import hydra
import torch
from robot_io.cams.realsense.realsense import Realsense  # noqa

from tacorl.utils.misc import log_rank_0
from tacorl.utils.networks import load_evaluation_checkpoint


@hydra.main(config_path="../config", config_name="evaluate_real_world")
def main(cfg):
    """
    Evaluate the policy in the panda robot

    """
    # Init module
    pl_module = load_evaluation_checkpoint(cfg).cuda()

    modalities = list(pl_module.all_modalities)
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot, modalities=modalities)

    # Init environment
    img = cv2.imread(cfg.img_path)
    goal = {}
    goal["rgb_static"] = img[:, :, ::-1]
    reset_info = {"goal": goal}

    # Init utils for eval
    transform_manager = hydra.utils.instantiate(cfg.transform_manager)
    rollout_manager = hydra.utils.instantiate(
        cfg.rollout_manager, transform_manager=transform_manager
    )

    # Start rollout
    with torch.no_grad():
        log_rank_0("Starting evaluation rollout ...")
        pl_module.eval()

        rollout_manager.episode_rollout(
            pl_module=pl_module,
            env=env,
            reset_info=reset_info,
            render=cfg.render,
        )
        pl_module.train()
        log_rank_0("Finished evaluation rollout")


if __name__ == "__main__":
    main()
