import numpy as np
import Sofa

from Pose_Transform.pose_transform import pose_to_position_moment


class Position_Estimate(Sofa.Core.Controller):

    def __init__(self, info, scene, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.info = info
        self.scene = scene
        self.p_pose = np.zeros((3, 1))
        self.h_hat_pose = np.zeros((3, 1))
        if not info[7] or info[7][0].get("algorithm_type") != "none":
            raise ValueError('Imitation-learned control uses direct capsule pose only: algorithm_type="none".')

    def onAnimateBeginEvent(self, event):
        self.p_pose, self.h_hat_pose = pose_to_position_moment(
            self.scene.instrument[0]["position_active"][:]
        )
