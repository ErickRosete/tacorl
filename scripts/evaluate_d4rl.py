import json
import logging
from pathlib import Path

import d4rl  # noqa
import gym
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from tacorl.utils.misc import log_rank_0
from tacorl.utils.networks import load_evaluation_checkpoint

log = logging.getLogger(__name__)


class EvaluationManager(object):
    def __init__(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        rollout_manager: DictConfig = None,
    ):
        self.pl_module = pl_module
        self.env = env
        self.rollout_manager = hydra.utils.instantiate(rollout_manager)

    @torch.no_grad()
    def evaluate_task(
        self,
        num_rollouts: int = 5,
        render: bool = False,
    ):
        log_rank_0(f"Evaluating D4RL env by performing {num_rollouts} rollouts")
        self.pl_module.eval()

        task_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "score": [],
            "succesful_episodes": 0,
        }

        for i in tqdm(range(num_rollouts)):
            info = self.rollout_manager.episode_rollout(
                pl_module=self.pl_module,
                env=self.env,
                render=render,
            )
            task_info["episode_returns"].append(info["episode_return"])
            task_info["episodes_lengths"].append(info["episode_length"])
            task_info["score"].append(info["score"])
            task_info["succesful_episodes"] += int(info["success"])

        task_info = {
            "accuracy": task_info["succesful_episodes"] / num_rollouts,
            "avg_episode_return": np.mean(task_info["episode_returns"]),
            "avg_episode_length": np.mean(task_info["episodes_lengths"]),
            "score": np.mean(task_info["score"]),
            "num_rollouts": num_rollouts,
        }
        log.info(task_info)
        self.pl_module.train()
        return task_info


@hydra.main(config_path="../config", config_name="evaluate_d4rl")
def main(cfg):
    pl_module = load_evaluation_checkpoint(cfg).cuda()
    env = gym.make(cfg.d4rl_env)
    eval_manager = EvaluationManager(pl_module=pl_module, env=env, **cfg.evaluation)
    model_results = eval_manager.evaluate_task(
        num_rollouts=cfg.num_rollouts, render=cfg.render
    )

    with open(cfg.filename, "w") as fp:
        json.dump(model_results, fp, indent=4)


if __name__ == "__main__":
    main()
