import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from Control_Package.ImitationLearnedPolicy import ILC_STATE_SIZE
from Magnetic_Engine.permanent_magnet import force_moment, force_moment_jacob
from Pose_Transform.pose_transform import position_moment_to_pose
from Sofa_Interface.sofa_interface import Sofa_Interface


sofa_interface = Sofa_Interface()
CONTROL_SIZE = 6


def _wrap_robot_theta(theta):
    return (np.asarray(theta, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _attitude_error_rotvec(current_pose, target_pose):
    current_rotation = R.from_quat(current_pose[3:7])
    target_rotation = R.from_quat(target_pose[3:7])
    return (target_rotation * current_rotation.inv()).as_rotvec()


def _capsule_tracking_state(scene, path, pc, v_pos, v_angle):
    target_pose = position_moment_to_pose(path.p_desired, path.h_hat_desired)
    attitude_error = _attitude_error_rotvec(scene.instrument[0]["position_active"][:], target_pose)
    return np.concatenate(
        (
            np.asarray(path.p_desired - pc, dtype=float).reshape(3),
            np.asarray(attitude_error, dtype=float).reshape(3),
            np.asarray(v_pos, dtype=float).reshape(3),
            np.asarray(v_angle, dtype=float).reshape(3),
        )
    )


def _capsule_ilc_state(scene, path, x):
    current_pose = np.asarray(scene.instrument[0]["position_active"][:], dtype=float).reshape(7)
    target_pose = np.asarray(position_moment_to_pose(path.p_desired, path.h_hat_desired), dtype=float).reshape(7)
    robot_theta = np.asarray(scene.magnetic_source[0]["theta"], dtype=float).reshape(6)
    magnet_position, magnet_direction = scene.magnetic_source[0]["robot"].fkine(
        scene.magnetic_source[0]["theta"]
    )
    state = np.concatenate(
        (
            np.asarray(x, dtype=float).reshape(12),
            current_pose[:3],
            current_pose[3:7],
            target_pose[:3],
            target_pose[3:7],
            robot_theta,
            np.asarray(magnet_position, dtype=float).reshape(3),
            np.asarray(magnet_direction, dtype=float).reshape(3),
        )
    )
    return state.reshape(ILC_STATE_SIZE)


class _ControlPath:

    def __init__(self, p_desired, h_hat_desired):
        self.p_desired = np.asarray(p_desired, dtype=float).reshape(3, 1)
        self.h_hat_desired = np.asarray(h_hat_desired, dtype=float).reshape(3, 1)


def _global_control_path(owner, path, pose_estimate, info):
    control_info = info[8][0]
    if not bool(control_info.get("global_recovery_enabled", False)):
        return path, "local"

    final_p = np.asarray(path.p_desired, dtype=float).reshape(3, 1)
    final_h = np.asarray(path.h_hat_desired, dtype=float).reshape(3, 1)
    pc = np.asarray(pose_estimate.p_pose, dtype=float).reshape(3, 1)
    final_error = float(np.linalg.norm(final_p - pc))
    final_switch_error = float(control_info.get("global_final_switch_error", 0.004))
    safe_z = float(control_info.get("global_safe_z", final_p[2, 0] + 0.022))
    safe_z = max(safe_z, final_p[2, 0] + float(control_info.get("global_safe_margin", 0.018)))
    trigger_z = float(control_info.get("global_trigger_z", final_p[2, 0] + 0.008))
    z_tol = float(control_info.get("global_z_tolerance", 0.0015))
    xy_tol = float(control_info.get("global_xy_tolerance", 0.004))
    reset_low_z = float(control_info.get("global_reset_low_z_error", 0.006))

    final_target = getattr(owner, "global_final_target", None)
    if final_target is None or np.linalg.norm(np.asarray(final_target).reshape(3, 1) - final_p) > 1e-6:
        use_recovery = final_error > final_switch_error and pc[2, 0] <= trigger_z
        owner.global_phase = "lift" if use_recovery else "final"
        owner.global_phase_steps = 0
        owner.global_final_target = final_p.copy()

    owner.global_phase_steps = int(getattr(owner, "global_phase_steps", 0)) + 1
    phase = getattr(owner, "global_phase", "lift")
    if phase == "final" and final_error > final_switch_error and pc[2, 0] <= min(trigger_z, final_p[2, 0] - reset_low_z):
        phase = "lift"
        owner.global_phase_steps = 0
    if phase == "lift" and pc[2, 0] >= safe_z - z_tol:
        phase = "transit"
        owner.global_phase_steps = 0
    if phase == "transit" and np.linalg.norm((final_p - pc)[:2]) <= xy_tol:
        phase = "final"
        owner.global_phase_steps = 0
    owner.global_phase = phase

    if phase == "lift":
        waypoint = np.array([[pc[0, 0]], [pc[1, 0]], [safe_z]])
    elif phase == "transit":
        waypoint = np.array([[final_p[0, 0]], [final_p[1, 0]], [safe_z]])
    else:
        waypoint = final_p
    return _ControlPath(waypoint, final_h), f"global_{phase}"


def _clip_force_moment(u, info):
    control_info = info[8][0]
    max_force = float(control_info.get("max_force", 0.0015))
    max_moment = float(control_info.get("max_moment", 8e-4))
    u = np.asarray(u, dtype=float).reshape(CONTROL_SIZE)
    u[:3] = np.clip(u[:3], -max_force, max_force)
    u[3:] = np.clip(u[3:], -max_moment, max_moment)
    return u


def _apply_direct_capsule_force(scene, u):
    u = np.asarray(u, dtype=float).reshape(CONTROL_SIZE)
    force = np.mat(u[:3]).transpose()
    moment = np.mat(u[3:]).transpose()
    sofa_interface.set_force_moment(scene.instrument[0]["force_torque_capsule"], force, moment)


def _robot_delta_from_force_moment(scene, pc, mc_hat, u):
    pa, ma_hat = scene.magnetic_source[0]["robot"].fkine(scene.magnetic_source[0]["theta"])
    ma_norm = scene.magnetic_source[0]["moment"]
    ma = ma_hat * ma_norm
    mc = mc_hat * scene.instrument[0]["moment"]
    p = pc - pa
    force, moment = force_moment(p, ma, mc)

    u = np.asarray(u, dtype=float).reshape(CONTROL_SIZE)
    force_desired = np.mat(u[:3]).transpose()
    moment_desired = np.mat(u[3:]).transpose()
    f_m_change = np.mat(np.concatenate((force_desired - force, moment_desired - moment), axis=0))

    jf_val = force_moment_jacob(p, ma, mc)[0:6, 0:6]
    jf_val[0:6, 3:6] = jf_val[0:6, 3:6] * ma_norm
    pc_change = np.mat(sofa_interface.get_velocity(scene.instrument[0]["velocity_active"])) * scene.dt
    d_change = f_m_change - jf_val * pc_change

    ja_val = scene.magnetic_source[0]["robot"].fkine_jacob(scene.magnetic_source[0]["theta"])
    source_to_capsule = np.mat(
        [
            [-1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    jfa = jf_val * source_to_capsule * ja_val
    theta_change = np.mat(np.linalg.pinv(jfa)) * d_change
    theta_change = np.clip(theta_change, -0.0008, 0.0008)
    return np.asarray(theta_change, dtype=float).reshape(6)


def _apply_robot_theta_change(scene, theta_change):
    theta_change = np.asarray(theta_change, dtype=float).reshape(scene.magnetic_source[0]["theta"].shape)
    scene.magnetic_source[0]["theta"] = _wrap_robot_theta(scene.magnetic_source[0]["theta"] + theta_change)
    pose_list = scene.magnetic_source[0]["robot"].fkine_all_link(scene.magnetic_source[0]["theta"])
    for i in range(len(scene.magnetic_source[0]["link_pose_list"])):
        scene.magnetic_source[0]["link_pose_list"][i].position[0][:] = pose_list[i]


def _update_robot_from_force_moment(scene, x, pc, mc_hat, u, info):
    control_info = info[8][0]
    stop_error_norm = float(control_info.get("robot_stop_error_norm", 0.0))
    attitude_stop_error_norm = float(control_info.get("attitude_stop_error_norm", np.deg2rad(1.0)))
    if (
        stop_error_norm > 0
        and np.linalg.norm(x[:3]) <= stop_error_norm
        and np.linalg.norm(x[3:6]) <= attitude_stop_error_norm
    ):
        return "stopped", 0.0

    theta_change = _robot_delta_from_force_moment(scene, pc, mc_hat, u)
    _apply_robot_theta_change(scene, theta_change)
    return "jacobian", float(np.linalg.norm(theta_change))


def _write_controller_pose_log(owner, mode, control_info, scene, path, i, u, robot_update_mode, robot_theta_delta_norm):
    if not control_info.get("pose_log_enabled", True):
        return
    interval = int(control_info.get("pose_log_interval", 1))
    if interval <= 0 or int(i) % interval != 0:
        return

    log_path = control_info.get("pose_log_path", "post_processing/ilc_pose_log.csv")
    if not getattr(owner, "pose_log_initialized", False):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "t,mode,capsule_x,capsule_y,capsule_z,capsule_qx,capsule_qy,capsule_qz,capsule_qw,"
                "target_x,target_y,target_z,target_qx,target_qy,target_qz,target_qw,"
                "error_x,error_y,error_z,error_norm,attitude_error_norm,force_norm,moment_norm,"
                "robot_update_mode,robot_theta_delta_norm\n"
            )
        owner.pose_log_initialized = True

    current_pose = np.asarray(scene.instrument[0]["position_active"][:], dtype=float).reshape(7)
    target_pose = np.asarray(position_moment_to_pose(path.p_desired, path.h_hat_desired), dtype=float).reshape(7)
    error_vec = target_pose[:3] - current_pose[:3]
    attitude_error = _attitude_error_rotvec(current_pose, target_pose)
    u = np.asarray(u, dtype=float).reshape(CONTROL_SIZE)
    values = [
        i / 100.0,
        mode,
        *current_pose.tolist(),
        *target_pose.tolist(),
        *error_vec.tolist(),
        float(np.linalg.norm(error_vec)),
        float(np.linalg.norm(attitude_error)),
        float(np.linalg.norm(u[:3])),
        float(np.linalg.norm(u[3:])),
        robot_update_mode,
        float(robot_theta_delta_norm),
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in values) + "\n")


def Create_Animate_Capsule_ILC(scene, i, path, pose_estimate, ilc_policy, info):
    pc = pose_estimate.p_pose
    v_pos = np.array([scene.instrument[0]["velocity_active"][0:3]]).transpose()
    v_angle = np.array([scene.instrument[0]["velocity_active"][3:6]]).transpose()

    control_path, global_mode = _global_control_path(ilc_policy, path, pose_estimate, info)
    x = _capsule_tracking_state(scene, control_path, pc, v_pos, v_angle)
    state = _capsule_ilc_state(scene, control_path, x)
    u = _clip_force_moment(ilc_policy.control(state), info)

    _apply_direct_capsule_force(scene, u)
    robot_update_mode, robot_theta_delta_norm = _update_robot_from_force_moment(
        scene, x, pc, pose_estimate.h_hat_pose, u, info
    )

    control_info = info[8][0]
    mode = ilc_policy.last_mode if global_mode == "local" else f"{ilc_policy.last_mode}_{global_mode}"
    _write_controller_pose_log(
        ilc_policy, mode, control_info, scene, path, i, u, robot_update_mode, robot_theta_delta_norm
    )

    print_interval = int(control_info.get("print_interval", 100))
    if print_interval > 0 and int(i) % print_interval == 0:
        current_pos = np.asarray(pc, dtype=float).reshape(3)
        target_pos = np.asarray(path.p_desired, dtype=float).reshape(3)
        error_vec = target_pos - current_pos
        angle_error = np.linalg.norm(
            _attitude_error_rotvec(
                scene.instrument[0]["position_active"][:],
                position_moment_to_pose(path.p_desired, path.h_hat_desired),
            )
        )
        print(
            "[MagRobotILCPosition] "
            f"mode={mode} "
            f"robot_update={robot_update_mode} "
            f"current=({current_pos[0]:.5f}, {current_pos[1]:.5f}, {current_pos[2]:.5f}) "
            f"target=({target_pos[0]:.5f}, {target_pos[1]:.5f}, {target_pos[2]:.5f}) "
            f"error=({error_vec[0]:.5f}, {error_vec[1]:.5f}, {error_vec[2]:.5f}) "
            f"err_norm={np.linalg.norm(error_vec):.5f} m "
            f"angle_err={angle_error:.5f} rad"
        )
