import Sofa

from Control_Package.ImitationLearnedPolicy import ImitationLearnedPolicy
from Module_Package.animate_module import Create_Animate_Capsule_ILC
from Sofa_Interface.sofa_interface import Sofa_Interface


class MagController(Sofa.Core.Controller):

    def __init__(self, root_node, info, simulator, scene, path, pose_estimate, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.simulator = simulator
        self.scene = scene
        self.root_node = root_node
        self.i = 0
        self.path = path
        self.pose_estimate = pose_estimate
        self.info = info

        if not info[8] or info[8][0].get("control_type") != "ilc":
            raise ValueError('This repository is configured for control_type="ilc" only.')
        if not info[1] or info[1][0].get("drive_source") != "robot_magnet":
            raise ValueError('Imitation-learned control requires drive_source="robot_magnet".')
        if not info[2] or info[2][0].get("object") != "capsule":
            raise ValueError('Imitation-learned control requires object="capsule".')

        self.ilc_policy = ImitationLearnedPolicy(info[8][0])
        self.sofa_interface = Sofa_Interface()

    def onAnimateBeginEvent(self, event):
        self.i += self.scene.dt * 100
        Create_Animate_Capsule_ILC(
            self.scene,
            self.i,
            self.path,
            self.pose_estimate,
            self.ilc_policy,
            self.info,
        )
        if self.info[5][0].get("cam", 0) == 1:
            pos = self.scene.instrument[0]["position_active"]
            self.simulator.cam.cameraRigid.value = [
                pos[0],
                pos[1],
                pos[2],
                0.707 * pos[3] - 0.707 * pos[5],
                0.707 * pos[4] + 0.707 * pos[6],
                0.707 * pos[3] + 0.707 * pos[5],
                -0.707 * pos[4] + 0.707 * pos[6],
            ]
