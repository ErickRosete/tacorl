import logging

import numpy as np
from gym import spaces

from tacorl.envs.rl_base_env import RLBaseEnv

log = logging.getLogger(__name__)


class PlayTableEnv(RLBaseEnv):
    def __init__(
        self, name: str = "play-table-v0", task: str = "open_drawer", **kwargs
    ):
        super(PlayTableEnv, self).__init__(**kwargs)
        self.name = name
        self.task = task
        self.max_distance = 0.5
        self.success_threshold = 0.95

    def get_observation_space(self):
        observation_space = super().get_observation_space()
        if "scene_obs" in self.modalities:
            observation_space.spaces["scene_obs"] = spaces.Box(
                low=0, high=1, shape=(1,)
            )
            self.scene_dim = 1
        return observation_space

    def get_scene_obs(self):
        return np.array([self.get_target_joint()])

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        return self.get_target_joint() > self.success_threshold

    def _reward(self):
        if self.sparse_reward:
            return int(self._success())
        else:
            target_joint = self.get_target_joint()
            dist = self.get_end_effector_handle_distance()
            dist = min(dist, self.max_distance)
            norm_dist = self._normalize(dist, 0, self.max_distance)
            # reward_near = 1 - norm_dist ** 0.4
            # positive, range [0, 1], bigger incentive near 1
            # reward_state = 1 - (1 - target_joint) ** 0.6
            # positive, range [0, 1], bigger incentive near 1
            reward_near = -norm_dist  # negative, range [-1, 0]
            reward_state = target_joint - 1  # negative, range [-1, 0]
            # reward_near = 1 - norm_dist # positive, range [0, 1]
            # reward_state = target_joint # positive, range [0, 1]
            # reward_success = int(self._success()) * 2
            # reward_success *= (self.max_episode_steps - self.current_step)
            reward = reward_near + reward_state
            info = {"reward_state": reward_state, "reward_near": reward_near}
        return reward, info

    def get_handle_position(self):
        """Get handle position of slide"""
        handle_position = np.array([0.0, 0.0, 0.0])
        for f_object in self.scene.fixed_objects:
            if "table" in f_object.name:
                uid = f_object.info_dict["uid"]
                if "slide" in self.task:
                    link = f_object.info_dict["links"]["slide_link"]
                    offset = np.array([0.275, -0.05, 0.01])
                elif "drawer" in self.task:
                    link = f_object.info_dict["links"]["drawer_link"]
                    offset = np.array([0, -0.185, 0])
            f_object_position = np.array(
                self.p.getLinkState(uid, link, physicsClientId=self.cid)[0]
            )
            handle_position = f_object_position + offset
            break

        # Debug point
        self.p.addUserDebugText("o", handle_position)
        return handle_position

    def get_end_effector_handle_distance(self):
        handle_position = self.get_handle_position()
        end_effector_position = self.robot.get_observation()[0][:3]
        return np.linalg.norm(handle_position - end_effector_position)

    def get_target_joint(self):
        """Get target joint value in range 0 to 1"""
        target_state = 0
        for door in self.scene.doors:
            if ("slide" in self.task and "slide" in door.name) or (
                "drawer" in self.task and "drawer" in door.name
            ):
                joint_limits = self.p.getJointInfo(
                    door.uid, door.joint_index, physicsClientId=self.cid
                )[8:10]
                target_state = self._normalize(
                    door.get_state(), joint_limits[0], joint_limits[1]
                )
                break
        # target_position = self.get_target_position()
        # self.p.addUserDebugText("%.02f" % target_state, target_position)
        return target_state
