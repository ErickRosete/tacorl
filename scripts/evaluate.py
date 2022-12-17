import json
import logging
from pathlib import Path

import gym
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from tacorl.utils.gym_utils import make_env
from tacorl.utils.misc import log_rank_0
from tacorl.utils.networks import load_pl_module_from_checkpoint

log = logging.getLogger(__name__)


class EvaluationManager(object):
    def __init__(
        self,
        pl_module: pl.LightningModule,
        env: gym.Env,
        transform_manager: DictConfig,
        single_task_rollout_generator: DictConfig = None,
        long_horizon_rollout_generator: DictConfig = None,
        long_horizon_seq_rollout_generator: DictConfig = None,
        rollout_manager: DictConfig = None,
    ):
        self.pl_module = pl_module
        self.env = env
        self.transform_manager = hydra.utils.instantiate(transform_manager)
        self.single_task_gen = hydra.utils.instantiate(single_task_rollout_generator)
        self.lh_task_gen = hydra.utils.instantiate(long_horizon_rollout_generator)
        self.lh_seq_task_gen = hydra.utils.instantiate(
            long_horizon_seq_rollout_generator
        )
        self.rollout_manager = hydra.utils.instantiate(
            rollout_manager, transform_manager=self.transform_manager
        )

    def evaluate_lh_tasks(
        self, filename="lh_tasks.json", render: bool = False, save_video: bool = False
    ):
        log_rank_0("Evaluating all long-horizon tasks")
        tasks_per_rollout = self.lh_task_gen.tasks_per_rollout
        success_acum_tasks = np.zeros(tasks_per_rollout)
        accum_len = []
        all_tasks_info = {}
        rollout_tasks = self.lh_task_gen.get_rollout_tasks()[:1000]
        for i, task in tqdm(enumerate(rollout_tasks)):
            task["completed_tasks"].sort()
            task_name = ("__").join(task["completed_tasks"])
            if task_name not in all_tasks_info:
                all_tasks_info[task_name] = []
            task_info = self.lh_rollout(
                task,
                render=render,
                save_video=save_video,
                rollout_name=f"rollout_lh_{i}",
            )
            task_info["successful_tasks"] = list(task_info["successful_tasks"])
            all_tasks_info[task_name].append(task_info)
            accum_len.append(len(task_info["successful_tasks"]))
            success_acum_tasks[: len(task_info["successful_tasks"])] += 1
            with open(filename, "w") as fp:
                json.dump(all_tasks_info, fp, indent=4)

        accuracy = success_acum_tasks / len(rollout_tasks)
        model_results = {}
        for i in range(len(accuracy)):
            model_results[f"lh_{i+1}_accuracy"] = accuracy[i].item()
        model_results.update(
            {
                "avg_len": np.mean(accum_len),
                "num_rollouts": len(rollout_tasks),
                "tasks_per_rollout": tasks_per_rollout,
                "tasks_info": all_tasks_info,
            }
        )
        with open(filename, "w") as fp:
            json.dump(model_results, fp, indent=4)

    @torch.no_grad()
    def lh_rollout(
        self,
        rollout_task,
        render: bool = False,
        save_video: bool = False,
        rollout_name: str = "rollout_lh",
    ):
        reset_info = {
            "task_info": {
                "start_info": self.lh_task_gen.get_state_info_from_step(
                    rollout_task["start_step"]
                ),
                "goal_info": self.lh_task_gen.get_state_info_from_step(
                    rollout_task["end_step"]
                ),
                "tasks": rollout_task["completed_tasks"],
            }
        }
        info = self.rollout_manager.episode_rollout(
            pl_module=self.pl_module,
            env=self.env,
            reset_info=reset_info,
            render=render,
            save_video=save_video,
            video_filename=f"{rollout_name}.mp4",
        )
        return info

    @torch.no_grad()
    def evaluate_all_tasks(
        self, filename="all_tasks.json", render: bool = False, save_video: bool = False
    ):
        log_rank_0("Evaluating all tasks")
        all_tasks_info = {}
        rollout_tasks = self.single_task_gen.get_rollout_tasks()
        for task_name, tasks in rollout_tasks.items():
            num_rollouts = min(len(tasks), 50)
            task_info = self.evaluate_task(
                task=task_name,
                num_rollouts=num_rollouts,
                render=render,
                save_video=save_video,
            )
            all_tasks_info[task_name] = task_info

            with open(filename, "w") as fp:
                json.dump(all_tasks_info, fp, indent=4)

    @torch.no_grad()
    def evaluate_task(
        self,
        task: str,
        num_rollouts: int = 5,
        render: bool = False,
        save_video: bool = False,
    ):

        log_rank_0(f"Evaluating task: {task}")
        self.pl_module.eval()

        task_info = {
            "episode_returns": [],
            "episodes_lengths": [],
            "succesful_episodes": 0,
        }

        rollouts_to_perform = min(
            num_rollouts, self.single_task_gen.get_num_rollouts_from_task(task=task)
        )
        for task_idx in tqdm(range(rollouts_to_perform)):
            reset_info = self.single_task_gen.get_reset_info(
                task=task, task_idx=task_idx
            )
            info = self.rollout_manager.episode_rollout(
                pl_module=self.pl_module,
                env=self.env,
                reset_info=reset_info,
                render=render,
                save_video=save_video,
                video_filename=f"{task}_{task_idx}.mp4",
            )
            task_info["episode_returns"].append(info["episode_return"])
            task_info["episodes_lengths"].append(info["episode_length"])
            task_info["succesful_episodes"] += int(info["success"])

        task_info = {
            "accuracy": task_info["succesful_episodes"] / rollouts_to_perform,
            "avg_episode_return": np.mean(task_info["episode_returns"]),
            "avg_episode_length": np.mean(task_info["episodes_lengths"]),
            "num_rollouts": rollouts_to_perform,
        }
        log.info(task)
        log.info(task_info)

        self.pl_module.train()
        return task_info

    @torch.no_grad()
    def evaluate_lh_seq_tasks(
        self,
        filename="lh_seq_tasks.json",
        render: bool = False,
        save_video: bool = False,
    ):
        """Evaluate long horizon tasks using intermediate goals"""

        log_rank_0("Evaluating all long-horizon sequential tasks")
        tasks_per_rollout = self.lh_seq_task_gen.tasks_per_rollout
        all_tasks_info = {"failed": {}, "success": {}}
        success_acum_tasks = np.zeros(tasks_per_rollout)
        rollout_tasks = self.lh_seq_task_gen.get_rollout_tasks()
        rollout_tasks = list(rollout_tasks.items())[:500]

        accum_len = []
        for rt_idx, task in tqdm(enumerate(rollout_tasks)):
            start_idx, end_tasks = task
            start_info = self.lh_seq_task_gen.get_state_info_from_step(int(start_idx))
            reset_info = {"task_info": {"start_info": start_info}}
            rollout_success_tasks = []
            for st_idx, (end_idx, evaluated_tasks) in enumerate(end_tasks.items()):
                reset_info["task_info"][
                    "goal_info"
                ] = self.lh_seq_task_gen.get_state_info_from_step(int(end_idx))
                info = self.rollout_manager.episode_rollout(
                    pl_module=self.pl_module,
                    env=self.env,
                    reset_info=reset_info,
                    render=render,
                    save_video=save_video,
                    video_filename=f"lh_seq_{rt_idx}_{st_idx}.mp4",
                )
                rollout_success_tasks.extend(list(info["successful_tasks"]))
                if "start_info" in reset_info["task_info"]:
                    del reset_info["task_info"]["start_info"]

            # Log info
            rollout_success_tasks = list(
                set(rollout_success_tasks).intersection(set(evaluated_tasks))
            )
            success_acum_tasks[: len(rollout_success_tasks)] += 1

            accum_len.append(len(rollout_success_tasks))
            for task in evaluated_tasks:
                if task in rollout_success_tasks:
                    if task not in all_tasks_info["success"]:
                        all_tasks_info["success"][task] = 1
                    else:
                        all_tasks_info["success"][task] += 1
                else:
                    if task not in all_tasks_info["failed"]:
                        all_tasks_info["failed"][task] = 1
                    else:
                        all_tasks_info["failed"][task] += 1

        accuracy = success_acum_tasks / len(rollout_tasks)
        model_results = {}
        for i in range(len(accuracy)):
            model_results[f"lh_{i+1}_accuracy"] = accuracy[i].item()
        model_results.update(
            {
                "avg_len": np.mean(accum_len),
                "num_rollouts": len(rollout_tasks),
                "tasks_per_rollout": tasks_per_rollout,
                "tasks_info": all_tasks_info,
            }
        )
        with open(filename, "w") as fp:
            json.dump(model_results, fp, indent=4)


@hydra.main(config_path="../config", config_name="evaluate")
def main(cfg):
    module_path = str(Path(cfg.module_path).expanduser())
    pl_module = load_pl_module_from_checkpoint(module_path).cuda()
    env = pl_module.env if hasattr(pl_module, "env") else make_env(cfg.env)
    eval_manager = EvaluationManager(pl_module=pl_module, env=env, **cfg.evaluation)
    if cfg.eval_type == "short_horizon":
        eval_manager.evaluate_all_tasks(cfg.filename, render=cfg.render)
    elif cfg.eval_type == "long_horizon":
        eval_manager.evaluate_lh_tasks(cfg.filename, render=cfg.render)
    elif cfg.eval_type == "long_horizon_sequential":
        eval_manager.evaluate_lh_seq_tasks(cfg.filename, render=cfg.render)


if __name__ == "__main__":
    main()
