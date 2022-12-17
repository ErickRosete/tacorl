import logging

import numpy as np
import torch.distributed as dist
import wandb
import wandb.util
from pytorch_lightning.loggers import WandbLogger

from tacorl.utils.misc import (
    add_img_thumbnail_to_imgs,
    delete_file,
    flatten,
    flatten_list_of_dicts,
)


class VideoLogger:
    """Custom Wandb Video Logger where only rank 0 can log videos"""

    def __init__(self, logger: WandbLogger):
        self.videos = []
        self.tasks = []
        self.video_paths = []
        self.logger = logger
        self.cons_logger = logging.getLogger(__name__)

    def new_video(self, initial_frame: np.ndarray, task: str = None) -> None:
        """
        Begin a new video with the first frame of a rollout.
        Args:
             initial_frame: static camera RGB image, shape: h, w, c
        """
        # h, w, c -> 1, c, h, w
        mod_init_frame = np.expand_dims(initial_frame.transpose(2, 0, 1), axis=0)
        self.videos.append(mod_init_frame)
        if task is not None:
            self.tasks.append(task)

    def update(self, img: np.ndarray) -> None:
        """
        Add new frame to video.
        Args:
            img: static camera RGB images, shape: h, w, c
        """
        # h, w, c -> 1, c, h, w
        mod_img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        # T, c, h, w
        self.videos[-1] = np.concatenate(
            [self.videos[-1], mod_img], axis=0
        )  # shape num_videos, t, c, h, w

    def add_goal_thumbnail(self, goal_img):
        """
        Add goal thumbnail to all stored images.
        Args:
            goal_img: static camera RGB images, shape: h, w, c
        """
        self.videos[-1] = add_img_thumbnail_to_imgs(self.videos[-1], goal_img)

    def write_to_tmp(self):
        """
        Save the videos as GIF in tmp directory,
        then log them at the end of the validation epoch from rank 0 process.
        """
        if len(self.tasks) > 0:
            assert len(self.tasks) == len(
                self.videos
            ), "The number of videos added doesn't match the number of registered tasks"
            if isinstance(self.video_paths, list):
                self.video_paths = {}
            for video, task in zip(self.videos, self.tasks):
                wandb_vid = wandb.Video(video, fps=10, format="gif")
                self.video_paths[task] = wandb_vid._path
        else:
            for video in self.videos:
                wandb_vid = wandb.Video(video, fps=10, format="gif")
                self.video_paths.append(wandb_vid._path)

        self.videos = []
        self.tasks = []

    def log_videos_to_wandb(self, name: str = "validation/rollout"):
        if dist.is_available() and dist.is_initialized():
            all_video_paths = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_video_paths, self.video_paths)
            if dist.get_rank() != 0:
                return
            if isinstance(all_video_paths[0], dict):
                video_paths = flatten_list_of_dicts(all_video_paths)
            else:
                video_paths = flatten(all_video_paths)
        else:
            video_paths = self.video_paths

        if isinstance(video_paths, dict):
            for task, path in video_paths.items():
                self.logger.experiment.log(
                    {f"{name}/{task}": wandb.Video(path, fps=10, format="gif")}
                )
                delete_file(path)
        else:
            for path in video_paths:
                self.logger.experiment.log(
                    {f"{name}": wandb.Video(path, fps=10, format="gif")}
                )
                delete_file(path)

    def log(self, name: str = "validation/rollout") -> None:
        """
        Call this method at the end of a validation epoch to log
        videos to wandb or filesystem.
        """
        if isinstance(self.logger, WandbLogger):
            self.log_videos_to_wandb(name)
        self.videos = []
        self.tasks = []
        self.video_paths = []
