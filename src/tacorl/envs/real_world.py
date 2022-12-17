from typing import List

import numpy as np
from robot_io.envs.robot_env import RobotEnv

MAX_REL_POS = 0.02
MAX_REL_ORN = 0.05


class RealWorld(RobotEnv):
    def __init__(
        self,
        modalities: List[str] = [],
        max_episode_steps: int = 500,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.modalities = modalities
        self._max_episode_steps = max_episode_steps

    def reset(
        self, goal, robot_obs=None, reset_to_neutral: bool = False, *args, **kwargs
    ):
        assert goal is not None, "Goal must not be empty"
        self.goal = goal
        if reset_to_neutral:
            return super().reset(*args, **kwargs)
        else:
            # Keep the robot in the same place
            if robot_obs is None:
                return self._get_obs()
            else:
                target_pos = robot_obs[:3]
                target_orn = robot_obs[3:6]
                gripper_state = "open" if robot_obs[-1] == 1 else "closed"
                return super().reset(
                    target_pos=target_pos,
                    target_orn=target_orn,
                    gripper_state=gripper_state,
                    *args,
                    **kwargs,
                )

    def _get_obs(self):
        """
        Get observation dictionary.
        Returns:
            Dictionary with observation and goal
        """
        obs = self.camera_manager.get_images()
        obs["robot_obs"] = self.robot.get_state()

        filtered_obs = {}
        for modality in self.modalities:
            filtered_obs[modality] = obs[modality].copy()

        observation = {"observation": filtered_obs, "goal": self.goal}
        return observation

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z. Angle in rad x, y, z. Gripper action
                    each value in range (-1, 1)
        """
        action = np.clip(action, -1.0, 1.0)
        new_action = {
            "motion": (
                action[:3] * MAX_REL_POS,
                action[3:6] * MAX_REL_ORN,
                1 if action[-1] > 0 else -1,
            ),
            "ref": "rel",
        }

        output = super().step(new_action)
        return output
