import numpy as np

from Module_Package.path_controller_module import PresetPathController, TargetTrajectoryVisual
from Pose_Transform.pose_transform import pose_to_position_moment


class Path:

    def __init__(self, root_node, info, scene):
        self.root_node = root_node
        self.dt = info[5][0]["step"]
        self.key = None

        if not info[6] or info[6][0].get("path_type") != "preset":
            raise ValueError('Imitation-learned control supports only path_type="preset".')

        self.p_desired, self.h_hat_desired = pose_to_position_moment(scene.instrument[0]["pose"])
        self.p_desired = np.asarray(self.p_desired, dtype=float).reshape(3, 1)
        self.h_hat_desired = np.asarray(self.h_hat_desired, dtype=float).reshape(3, 1)

        preset_controller = PresetPathController(
            self.p_desired,
            self.h_hat_desired,
            self.dt,
            info,
            scene,
            name="preset_path_controller",
        )
        root_node.addObject(preset_controller)

        if info[6][0].get("show_target_trajectory", True):
            target_visual = TargetTrajectoryVisual(
                root_node,
                self.p_desired,
                self.h_hat_desired,
                self.dt,
                info,
                name="target_trajectory_visual_controller",
            )
            root_node.addObject(target_visual)
