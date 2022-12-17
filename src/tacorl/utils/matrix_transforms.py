import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")


def matrix_to_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(
        position: np.array of shape (3,),
        orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    return R.from_matrix(mat[:3, :3]).as_quat()


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler("xyz", orn).as_matrix()
    mat[:3, 3] = pos
    return mat


def orn_to_matrix(orn):
    mat = np.eye(3)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler("xyz", orn).as_matrix()
    return mat


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def to_tcp_frame(rel_action_pos, rel_action_orn, tcp_orn):
    T_tcp_world = np.linalg.inv(orn_to_matrix(tcp_orn))
    pos_tcp_rel = T_tcp_world @ rel_action_pos
    T_world_tcp = orn_to_matrix(tcp_orn)
    T_world_tcp_new = orn_to_matrix(rel_action_orn) @ T_world_tcp

    orn_tcp_rel = quat_to_euler(
        matrix_to_orn(np.linalg.inv(T_world_tcp) @ T_world_tcp_new)
    )
    return pos_tcp_rel, orn_tcp_rel


def to_world_frame(rel_action_pos, rel_action_orn, tcp_orn):
    T_world_tcp_old = orn_to_matrix(tcp_orn)
    pos_w_rel = T_world_tcp_old[:3, :3] @ rel_action_pos
    T_tcp_new_tcp_old = orn_to_matrix(rel_action_orn)

    T_world_tcp_new = T_world_tcp_old @ np.linalg.inv(T_tcp_new_tcp_old)
    orn_w_rel = quat_to_euler(
        matrix_to_orn(T_world_tcp_old @ np.linalg.inv(T_world_tcp_new))
    )
    return pos_w_rel, orn_w_rel
