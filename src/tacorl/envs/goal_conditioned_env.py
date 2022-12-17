import logging
from typing import Dict, List

import cv2
import hydra
import numpy as np
from gym import spaces
from omegaconf import DictConfig

from tacorl.envs.rl_base_env import RLBaseEnv

log = logging.getLogger(__name__)


class GoalConditionedEnv(RLBaseEnv):
    def __init__(
        self,
        name: str = "goal-conditioned-v0",
        tasks: dict = {},
        initial_and_goal_states: dict = {},
        goal_modalities: List[str] = [],
        **kwargs
    ):

        self.goal_modalities = goal_modalities
        super(GoalConditionedEnv, self).__init__(**kwargs)
        self.name = name
        self.initial_and_goal_states = initial_and_goal_states
        self.tasks = hydra.utils.instantiate(tasks)
        # Init unset variables
        self.selected_tasks = []
        self.goal = None

    def get_possible_tasks(self):
        """Returns a dictionary with the possible tasks,
        each key represent a possible task and the value is
        the number of configurations available per task"""
        possible_tasks = {}
        for task, configs in self.initial_and_goal_states.items():
            possible_tasks[task] = len(configs)
        return possible_tasks

    def set_tasks(self, task_info: DictConfig = None):
        if "index" in task_info:
            return self.set_task_from_index(**task_info)
        elif "start_info" in task_info and "goal_info" in task_info:
            return self.set_tasks_from_complete_info(**task_info)
        elif "goal_info" in task_info:
            return self.set_tasks_from_goal_info(**task_info)
        else:
            raise ValueError("Invalid keys in task_info")

    def set_tasks_from_complete_info(
        self,
        tasks: List[str] = [],
        start_info: Dict[str, dict] = None,
        goal_info: Dict[str, dict] = None,
    ):
        """Setting task completely from task info"""
        super().reset(**goal_info)
        self.goal = super().get_obs(modalities=self.goal_modalities)
        end_info = self.get_info()
        obs = super().reset(**start_info)
        self.start_info = self.get_info()
        if tasks is None or len(tasks) == 0:
            self.selected_tasks = list(
                self.tasks.get_task_info(start_info=self.start_info, end_info=end_info)
            )
        else:
            self.selected_tasks = tasks
        return obs

    def set_task_from_index(self, task: str = None, index: int = 0):
        assert (
            task in self.initial_and_goal_states.keys()
        ), "task is not present in the list of saved initial and goal states"
        assert len(self.initial_and_goal_states[task]) > index, "invalid task index"

        self.selected_tasks = [task]
        init_and_goal = self.initial_and_goal_states[task][index]
        super().reset(
            robot_obs=np.asarray(init_and_goal["goal"]["robot_obs"]),
            scene_obs=np.asarray(init_and_goal["goal"]["scene_obs"]),
        )
        self.goal = super().get_obs(modalities=self.goal_modalities)

        # reset to corresponding initial scene
        obs = super().reset(
            robot_obs=np.asarray(init_and_goal["initial"]["robot_obs"]),
            scene_obs=np.asarray(init_and_goal["initial"]["scene_obs"]),
        )
        self.start_info = self.get_info()
        return obs

    def set_tasks_from_goal_info(
        self,
        goal_info: Dict[str, dict] = None,
    ):
        curr_state = self.get_state_obs()
        super().reset(**goal_info)
        self.goal = super().get_obs(modalities=self.goal_modalities)
        end_info = self.get_info()
        obs = super().reset(**curr_state)
        self.start_info = self.get_info()
        self.selected_tasks = list(
            self.tasks.get_task_info(start_info=self.start_info, end_info=end_info)
        )
        return obs

    def render(self, mode="human"):
        """render is gym compatibility function"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if mode == "human":
            if "rgb_static" not in rgb_obs:
                log.warning("Environment does not have static camera")
                return
            img = rgb_obs["rgb_static"][:, :, ::-1]
            size = img.shape[:2]
            i_h = int(size[0] / 3)
            i_w = int(size[1] / 3)
            resize_goal_img = cv2.resize(
                self.goal["rgb_static"][:, :, ::-1],
                dsize=(i_h, i_w),
                interpolation=cv2.INTER_CUBIC,
            )
            img[-i_h:, :i_w] = resize_goal_img

            cv2.imshow("simulation cam", cv2.resize(img, (500, 500)))
            cv2.waitKey(1)
        elif mode == "rgb_array":
            assert "rgb_static" in rgb_obs, "Environment does not have static camera"
            return rgb_obs["rgb_static"]
        else:
            raise NotImplementedError

    def reset(
        self, robot_obs=None, scene_obs=None, task_info: dict = None, *args, **kwargs
    ):
        if robot_obs is not None or scene_obs is not None:
            self.selected_tasks = []
            self.goal = None
            obs = super().reset(
                robot_obs=robot_obs, scene_obs=scene_obs, *args, **kwargs
            )
            self.start_info = self.get_info()
            return obs

        if task_info is not None:
            return self.set_tasks(task_info)

        # Sample a random task configuration
        random_task = np.random.choice(list(self.initial_and_goal_states.keys()))
        task_info = {
            "task": random_task,
            "index": np.random.choice(len(self.initial_and_goal_states[random_task])),
        }
        return self.set_tasks(task_info)

    def set_camera_views_size(self):
        self.views_size = {}
        for camera in self.cameras:
            rgb_cam_name = "rgb_%s" % camera.name
            if rgb_cam_name in self.modalities or rgb_cam_name in self.goal_modalities:
                self.views_size[rgb_cam_name] = (camera.height, camera.width)
            depth_cam_name = "depth_%s" % camera.name
            if (
                depth_cam_name in self.modalities
                or depth_cam_name in self.goal_modalities
            ):
                self.views_size[depth_cam_name] = (camera.height, camera.width)

    def get_observation_space(self):
        return spaces.Dict(
            {
                "observation": self.construct_observation_space_from_modalities(
                    self.modalities
                ),
                "goal": self.construct_observation_space_from_modalities(
                    self.goal_modalities
                ),
            }
        )

    def get_successful_tasks(self) -> List[str]:
        current_info = self.get_info()
        successful_tasks = self.tasks.get_task_info_for_set(
            self.start_info, current_info, self.selected_tasks
        )
        return successful_tasks

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        if len(self.selected_tasks) == 0:
            return False
        successful_tasks = self.get_successful_tasks()
        return set(self.selected_tasks) == set(successful_tasks)

    def get_obs(self):
        """Collect camera, robot and scene observations."""
        observation = {"observation": super().get_obs(), "goal": self.goal}
        return observation

    def _reward(self):
        reward = int(self._success())
        info = {"reward": reward, "successful_tasks": self.get_successful_tasks()}
        return reward, info
