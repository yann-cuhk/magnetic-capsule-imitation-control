import numpy as np

from Environment_Package.environment import soft_object, visu_object
from Module_Package.scene_module import Create_Magnetic_sphere, Create_Manipulator


class Scene:

    def __init__(self, root_node, info):
        self.dt = info[5][0]["step"]
        self.root_node = root_node
        self.instrument = []
        self.magnetic_source = []
        self.sensor = []
        self.show_pose = None
        self.pose = None
        self.current = None
        self.current_number = 0

        self._create_capsule(info)
        self._create_robot_magnet(info)
        self._create_environment(info)

    def _create_capsule(self, info):
        if len(info[2]) != 1 or info[2][0].get("object") != "capsule":
            raise ValueError('Imitation-learned control supports exactly one object="capsule".')

        capsule_info = info[2][0]
        capsule = {
            "position_active": None,
            "velocity_active": None,
            "force_torque_capsule": None,
            "capsule_Mechanical": None,
            "CFF_visu": None,
            "CFF_visu_2": None,
            "pose": capsule_info["pose"],
            "object": "capsule",
            "moment": capsule_info["moment"],
            "totalMass": capsule_info["capsule_mass"],
            "flotage": capsule_info["flotage"],
            "fluid_damping": capsule_info.get("fluid_damping", 0.0),
            "volume": 4.18879e-6,
            "inertiaMatrix": capsule_info["inertiaMatrix"],
        }
        (
            capsule["position_active"],
            capsule["capsule_Mechanical"],
            capsule["velocity_active"],
            capsule["force_torque_capsule"],
            capsule["CFF_visu"],
            capsule["CFF_visu_2"],
        ) = Create_Magnetic_sphere(
            self.root_node,
            capsule["pose"],
            capsule["totalMass"],
            capsule["volume"],
            capsule["inertiaMatrix"],
            info,
            flotage=capsule["flotage"],
            fluid_damping=capsule["fluid_damping"],
        )
        capsule["buoyancy_in_model"] = True
        self.instrument.append(capsule)

    def _create_robot_magnet(self, info):
        if len(info[1]) != 1 or info[1][0].get("drive_source") != "robot_magnet":
            raise ValueError('Imitation-learned control supports exactly one drive_source="robot_magnet".')
        source_info = info[1][0]
        if "magnet_pose" not in source_info:
            raise ValueError('Imitation-learned control requires an explicit "magnet_pose".')

        source = {
            "robot": None,
            "theta": None,
            "link_pose_list": [],
            "moment": source_info["moment"],
            "base_pose": source_info["manipulator_base_pose"],
            "magnet_pose": source_info["magnet_pose"],
            "visual": source_info["visual"],
        }
        source["robot"], source["theta"], source["link_pose_list"] = Create_Manipulator(
            self.root_node,
            source["base_pose"],
            source["magnet_pose"],
            source["visual"],
            X0=np.array([1.0, -1.0, -2.0, 1.0, 1.0, 1.0]),
        )
        self.magnetic_source.append(source)

    def _create_environment(self, info):
        for env_info in info[0]:
            env_type = env_info.get("env")
            if env_type == "soft_static":
                soft_object(
                    self.root_node,
                    path=env_info["surfaceMeshFilePath"],
                    volume_path=env_info["volumeMeshFilePath"],
                    pose=env_info["trans_env"],
                    totalMass=env_info["mass"],
                    poissonRatio=env_info["poissonRatio"],
                    youngModulus=env_info["youngModulus"],
                    scale=env_info["scale"],
                    color=env_info["color"],
                    isAStaticObject=True,
                    boxroi=env_info["indices"],
                )
            elif env_type == "visu":
                visu_object(
                    self.root_node,
                    path=env_info["file_path"],
                    pose=env_info["trans_env"],
                    scale=env_info["scale"],
                    color=env_info["color"],
                )
            else:
                raise ValueError(f'Unsupported ILC environment type: "{env_type}".')
