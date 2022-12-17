import logging

import numpy as np
import pybullet as p
from calvin_env.robot.robot import Robot

# A logger for this file
log = logging.getLogger(__name__)


class RLRobot(Robot):
    def relative_to_absolute(self, action):
        """Reimplementation trying to make actions more markovian"""
        assert len(action) == 7
        rel_pos, rel_orn, gripper = np.split(action, [3, 6])
        rel_pos *= self.max_rel_pos * self.magic_scaling_factor_pos
        rel_orn *= self.max_rel_orn * self.magic_scaling_factor_orn
        if self.use_target_pose:
            target_pos = self.target_pos + rel_pos
            target_orn = self.target_orn + rel_orn
            tcp_pos, tcp_orn = p.getLinkState(
                self.robot_uid, self.tcp_link_id, physicsClientId=self.cid
            )[:2]
            tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            tcp_pos, tcp_orn = np.array(tcp_pos), np.array(tcp_orn)
            self.target_pos = np.clip(
                target_pos, tcp_pos - self.max_rel_pos, tcp_pos + self.max_rel_pos
            )
            self.target_orn = np.clip(
                target_orn, tcp_orn - self.max_rel_orn, tcp_orn + self.max_rel_orn
            )
            return self.target_pos, self.target_orn, gripper
        else:
            tcp_pos, tcp_orn = p.getLinkState(
                self.robot_uid, self.tcp_link_id, physicsClientId=self.cid
            )[:2]
            tcp_orn = p.getEulerFromQuaternion(tcp_orn)
            abs_pos = np.array(tcp_pos) + rel_pos
            abs_orn = np.array(tcp_orn) + rel_orn
            return abs_pos, abs_orn, gripper
