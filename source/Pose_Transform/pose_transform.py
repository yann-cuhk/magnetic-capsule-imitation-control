import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def _column3(value):
    return np.asarray(value, dtype=float).reshape(3, 1)


def _rotation_from_x_to_vector(moment):
    direction = _column3(moment).reshape(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("Moment direction must be nonzero.")
    direction = direction / norm
    x_axis = np.array([1.0, 0.0, 0.0])
    dot = float(np.clip(np.dot(x_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        return R.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0])).as_matrix()
    axis = np.cross(x_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    return R.from_rotvec(axis * angle).as_matrix()


def moment_to_angle(m):
    m = _column3(m)
    phi = math.atan2(math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2), m[2, 0])
    theta = math.atan2(m[1, 0], m[0, 0])
    return phi, theta


def angle_to_axis_radian(phi, theta):
    direction = np.array([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)])
    x_axis = np.array([1.0, 0.0, 0.0])
    dot = float(np.clip(np.dot(x_axis, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-12:
        return math.pi, np.array([0.0, 0.0, 1.0])
    axis = np.cross(x_axis, direction)
    axis = axis / np.linalg.norm(axis)
    return math.acos(dot), axis


def axis_radian_to_rmatrix(angle, axis):
    axis = np.asarray(axis, dtype=float).reshape(3)
    if abs(angle) < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()


def rmatrix_to_quat(rot_matrix):
    return list(R.from_matrix(np.asarray(rot_matrix, dtype=float).reshape(3, 3)).as_quat())


def quat_to_rmatrix(quat):
    return R.from_quat(quat).as_matrix()


def rot_to_euler(rot_matrix, degree_mode=1):
    euler = R.from_matrix(np.asarray(rot_matrix, dtype=float).reshape(3, 3)).as_euler("xyz", degrees=bool(degree_mode))
    return np.array(euler)


def position_moment_to_pose(position, moment):
    pose = []
    position = _column3(position)
    pose.extend(list(position.transpose()[0]))
    pose.extend(rmatrix_to_quat(_rotation_from_x_to_vector(moment)))
    return pose


def T_matrix_to_pose(T_matrix):
    T_matrix = np.asarray(T_matrix, dtype=float).reshape(4, 4)
    pose = []
    pose.extend(list(T_matrix[0:3, 3:4].transpose()[0]))
    pose.extend(rmatrix_to_quat(T_matrix[0:3, 0:3]))
    return pose


def pose_to_position_moment(pose):
    position = np.asarray(pose[0:3], dtype=float).reshape(3, 1)
    rot_matrix = quat_to_rmatrix(pose[3:7])
    moment = rot_matrix @ np.array([[1.0], [0.0], [0.0]])
    return position, moment


def pose_to_T_matrix(pose):
    T = np.zeros((4, 4))
    T[0:3, 3:4] = np.asarray(pose[0:3], dtype=float).reshape(3, 1)
    T[0:3, 0:3] = quat_to_rmatrix(pose[3:7])
    T[3, 3] = 1
    return T
