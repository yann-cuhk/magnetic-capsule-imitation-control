import numpy as np
import Sofa

from Pose_Transform.pose_transform import position_moment_to_pose


STOMACH_SMALL_WAYPOINTS = [
    np.array([[0.000], [0.000], [0.000]]),
    np.array([[0.018], [-0.022], [-0.002]]),
    np.array([[0.040], [-0.052], [-0.005]]),
    np.array([[0.033], [-0.087], [-0.008]]),
    np.array([[0.000], [-0.112], [-0.010]]),
]


def _stomach_small_point(time, p_initial, segment_time):
    if time <= 0:
        return p_initial + STOMACH_SMALL_WAYPOINTS[0]
    segment = min(int(time / segment_time), len(STOMACH_SMALL_WAYPOINTS) - 2)
    local_t = (time - segment * segment_time) / segment_time
    local_t = min(max(local_t, 0.0), 1.0)
    smooth = 3 * local_t ** 2 - 2 * local_t ** 3
    return (
        p_initial
        + STOMACH_SMALL_WAYPOINTS[segment] * (1 - smooth)
        + STOMACH_SMALL_WAYPOINTS[segment + 1] * smooth
    )


def _normalized(vector):
    vector = np.asarray(vector, dtype=float).reshape(3, 1)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Target direction must be nonzero.")
    return vector / norm


class TargetTrajectoryVisual(Sofa.Core.Controller):

    def __init__(self, root_node, p_desired, h_hat_desired, dt, info, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.p_desired = p_desired
        self.h_hat_desired = h_hat_desired
        self.dt = dt
        self.info = info
        visual_info = info[6][0].get("visual", {}) if info[6] else {}
        sample_count = int(visual_info.get("samples", 180))
        duration = float(visual_info.get("duration", 30.0))
        self.path_points = self._sample_path(sample_count, duration)

        visual_node = root_node.addChild("target_trajectory_visual")
        path_node = visual_node.addChild("target_path_points_node")
        self.path_mo = path_node.addObject(
            "MechanicalObject",
            name="target_path_points",
            template="Vec3d",
            position=self.path_points,
            showObject=False,
            showObjectScale=float(visual_info.get("path_point_size", 0.004)),
            showColor=visual_info.get("path_color", [1.0, 0.85, 0.0, 1.0]),
        )
        target_node = visual_node.addChild("current_target_node")
        self.target_mo = target_node.addObject(
            "MechanicalObject",
            name="current_target",
            template="Rigid3",
            position=position_moment_to_pose(self.p_desired, self.h_hat_desired),
            showObject=True,
            showObjectScale=float(visual_info.get("target_size", 0.012)),
            showColor=visual_info.get("target_color", [0.0, 1.0, 0.0, 1.0]),
        )

    def _sample_path(self, sample_count, duration):
        sample_count = max(sample_count, 2)
        path_info = self.info[6][0]
        path_type = path_info.get("type")
        p_initial = self.p_desired.copy()

        if path_type == "stomach_small":
            segment_time = float(path_info.get("segment_time", 5.0))
            return [
                [
                    float(point[0, 0]),
                    float(point[1, 0]),
                    float(point[2, 0]),
                ]
                for point in (
                    _stomach_small_point(float(t), p_initial, segment_time)
                    for t in np.linspace(0, duration, sample_count)
                )
            ]
        if path_type == "static_target":
            target_position = path_info.get("target_position")
            if target_position is None:
                point = p_initial
            else:
                point = np.asarray(target_position, dtype=float).reshape(3, 1)
            return [[float(point[0, 0]), float(point[1, 0]), float(point[2, 0])]]
        raise ValueError(f'Unsupported ILC preset path type: "{path_type}".')

    def onAnimateBeginEvent(self, event):
        self.target_mo.position[0][:] = position_moment_to_pose(self.p_desired, self.h_hat_desired)


class PresetPathController(Sofa.Core.Controller):

    def __init__(self, p_desired, h_hat_desired, dt, info, scene, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.p_desired = p_desired
        self.h_hat_desired = h_hat_desired
        self.p_initial = p_desired.copy()
        self.h_hat_initial = h_hat_desired.copy()
        self.elapsed = 0.0
        self.dt = dt
        self.info = info
        self.scene = scene
        self.path_type = self.info[6][0].get("type")
        if self.path_type not in ("stomach_small", "static_target"):
            raise ValueError(f'Unsupported ILC preset path type: "{self.path_type}".')

    def onAnimateBeginEvent(self, event):
        self.elapsed += self.dt
        path_info = self.info[6][0]

        if self.path_type == "stomach_small":
            final_offset = STOMACH_SMALL_WAYPOINTS[-1]
            if path_info.get("target_mode") == "smooth":
                segment_time = float(path_info.get("segment_time", 3.0))
                self.p_desired[0:3] = _stomach_small_point(self.elapsed, self.p_initial, segment_time)
            else:
                self.p_desired[0:3] = self.p_initial + final_offset
            self.h_hat_desired[0:3] = _normalized(final_offset)
            return

        target_position = path_info.get("target_position")
        target_direction = path_info.get("target_direction")
        if target_position is not None:
            self.p_desired[0:3] = np.asarray(target_position, dtype=float).reshape(3, 1)
        if target_direction is not None:
            self.h_hat_desired[0:3] = _normalized(target_direction)
