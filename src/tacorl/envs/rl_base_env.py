import logging
from typing import List

import cv2
import numpy as np
from calvin_env.envs.play_table_env import PlayTableSimEnv
from gym import spaces

from tacorl.utils.egl import set_current_egl_device
from tacorl.utils.matrix_transforms import to_world_frame

logger = logging.getLogger(__name__)


class RLBaseEnv(PlayTableSimEnv):
    def __init__(
        self,
        sparse_reward: bool = False,
        max_episode_steps: int = 200,
        end_effector_pose: bool = False,
        modalities: List[str] = [],
        action_type: str = "rel_world",
        **kwargs,
    ):
        if "use_egl" in kwargs and kwargs["use_egl"]:
            set_current_egl_device()
        super(RLBaseEnv, self).__init__(**kwargs)
        assert (
            len(modalities) > 0
        ), "There must be at least one modality in the observation"

        self.action_type = action_type
        self.modalities = modalities
        self.set_camera_views_size()

        self.end_effector_pose = end_effector_pose
        self.max_episode_steps = max_episode_steps
        self.sparse_reward = sparse_reward

        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = self.get_observation_space()

    def set_camera_views_size(self):
        self.views_size = {}
        for camera in self.cameras:
            if "rgb_%s" % camera.name in self.modalities:
                self.views_size["rgb_%s" % camera.name] = (camera.height, camera.width)
            if "depth_%s" % camera.name in self.modalities:
                self.views_size["depth_%s" % camera.name] = (
                    camera.height,
                    camera.width,
                )

    def reset(self, *args, **kwargs):
        self.current_step = 0
        return super().reset(*args, **kwargs)

    def construct_observation_space_from_modalities(self, modalities: List[str] = []):
        observation_space = {}
        for view in modalities:
            if view == "scene_obs":
                observation_space[view] = spaces.Box(low=0, high=1, shape=(24,))
                self.scene_dim = 24
            elif view == "robot_obs":
                robot_obs_space = (
                    spaces.Box(low=-1, high=1, shape=(7,))
                    if self.end_effector_pose
                    else spaces.Box(low=-3.14, high=3.14, shape=(7,))
                )
                observation_space[view] = robot_obs_space
                self.proprioceptive_dim = 7
            else:
                observation_space[view] = spaces.Box(
                    low=-1,
                    high=1,
                    shape=(1, self.views_size[view][0], self.views_size[view][1]),
                )
        return spaces.Dict(observation_space)

    def get_observation_space(self):
        return self.construct_observation_space_from_modalities(self.modalities)

    def get_camera_obs_dict(self, modalities=None):
        assert self.cameras is not None

        if modalities is None:
            modalities = self.modalities

        camera_obs = {}
        if len(modalities) > 0:
            for cam in self.cameras:
                rgb_cam = "rgb_%s" % cam.name
                rgb_in_obs = rgb_cam in modalities
                depth_cam = "depth_%s" % cam.name
                depth_in_obs = depth_cam in modalities
                if rgb_in_obs or depth_in_obs:
                    rgb, depth = cam.render()
                    if rgb_in_obs:
                        camera_obs[rgb_cam] = rgb
                    if depth_in_obs:
                        camera_obs[depth_cam] = depth
        return camera_obs

    def get_scene_obs(self):
        return self.scene.get_obs()

    def get_obs(self, modalities=None):
        """Collect camera, robot and scene observations."""
        if modalities is None:
            modalities = self.modalities

        obs = self.get_camera_obs_dict(modalities=modalities)
        if "scene_obs" in modalities:
            obs["scene_obs"] = self.get_scene_obs()
        if "robot_obs" in modalities:
            robot_obs, robot_info = self.robot.get_observation()
            if self.end_effector_pose:
                robot_obs = robot_obs[:7]
            else:
                robot_obs = np.array(robot_info["arm_joint_states"])
            obs["robot_obs"] = robot_obs
        return obs

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        return False

    def _out_of_limits(self):
        """Returns a boolean indicating if the robot is out of bounds"""
        return False

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        out_of_limits = self._out_of_limits()
        done = success or out_of_limits
        d_info = {"success": success, "out_of_limits": out_of_limits}
        return done, d_info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z. Angle in rad x, y, z. Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
            observation: dict with keys:
                        rgb_obs
                        depth_obs
                        robot_obs
                        scene_obs
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1

        # Transform action to absolute action
        _, robot_info = self.robot.get_observation()
        if self.action_type == "abs":
            abs_action = env_action
        elif self.action_type == "rel_world":
            abs_action = self.robot.relative_to_absolute(env_action)
        elif self.action_type == "rel_tcp":
            pos_w, orn_w = to_world_frame(
                rel_action_pos=env_action[:3] * self.robot.max_rel_pos,
                rel_action_orn=env_action[3:6] * self.robot.max_rel_orn,
                tcp_orn=robot_info["tcp_orn"],
            )
            rel_action_world = np.concatenate(
                [
                    pos_w / self.robot.max_rel_pos,
                    orn_w / self.robot.max_rel_orn,
                    env_action[6:],
                ]
            )
            abs_action = self.robot.relative_to_absolute(rel_action_world)

        # Keep applying action until it reaches desired position or it stops moving
        curr_pos, last_pos = np.array(robot_info["tcp_pos"]), abs_action[0]
        perf_actions = 0
        while perf_actions == 0 or (
            perf_actions < 4
            and np.linalg.norm(abs_action[0] - curr_pos) > 0.005
            and np.linalg.norm(last_pos - curr_pos) > 0.005
        ):
            self.robot.apply_action(abs_action)
            for _ in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
            last_pos = curr_pos
            _, robot_info = self.robot.get_observation()
            curr_pos = np.array(robot_info["tcp_pos"])
            perf_actions += 1

        # Get new obs
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        self.current_step += 1
        return obs, reward, done, info

    def _normalize(self, val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    def debug_point(self, position):
        """Draw a circle in an rgb camera on a certain world coordinates
        Args
            position: 3D coordinates of world position where the circle will be drawn
        """
        radius = 5
        color = (255, 0, 0)
        hom_coord = np.append(position, 1)
        for cam in self.cameras:
            if "static" in cam.name:
                rgb, _ = cam.render()
                pixel_coords = cam.project(hom_coord)
                img = rgb[:, :, ::-1].astype(np.uint8).copy()
                img = cv2.circle(img, pixel_coords, radius, color, thickness=-1)
                cv2.imshow(cam.name, img)
                cv2.waitKey(1)
