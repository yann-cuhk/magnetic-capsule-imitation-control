import numpy as np

from Environment_Package.environment import visu_object
from Manipulator_Kinematics.kinematics import Manipulator2
from Pose_Transform.pose_transform import pose_to_position_moment


_MANIPULATOR_COUNT = 0


def Create_Magnetic_sphere(root_node, pose, totalMass, volume, inertiaMatrix, info, flotage=0.0, fluid_damping=0.0):
    capsule = root_node.addChild("magnetic_sphere")
    pose_capsule = capsule.addObject(
        "MechanicalObject",
        name="mstate",
        template="Rigid3",
        position=pose,
        showObject=True,
        showObjectScale=0.02,
    )
    capsule.addObject("UniformMass", name="mass", vertexMass=[totalMass, volume, inertiaMatrix[:]])
    capsule.addObject("EulerImplicitSolver", name="odesolver")
    capsule.addObject("SparseLDLSolver", name="solver", template="CompressedRowSparseMatrixd")
    capsule.addObject("LinearSolverConstraintCorrection")
    if fluid_damping:
        capsule.addObject(
            "UniformVelocityDampingForceField",
            name="fluid_damping",
            dampingCoefficient=float(fluid_damping),
            implicit=True,
        )
    force_torque_capsule = capsule.addObject(
        "ConstantForceField",
        name="ss",
        indices=[0],
        forces=[[0, 0, 0, 0, 0, 0]],
        indexFromEnd=True,
        showArrowSize="0",
    )
    capsule.addObject(
        "ConstantForceField",
        name="buoyancy",
        indices=[0],
        forces=[[0, 0, float(flotage), 0, 0, 0]],
        indexFromEnd=True,
        showArrowSize="0",
    )
    CFF_visu = capsule.addObject(
        "ConstantForceField",
        indices=[0],
        name="ceshi",
        forces=[[0, 0, 0, 0, 0, 0]],
        indexFromEnd=True,
        showArrowSize=info[5][0]["force_scale"][0],
    )
    CFF_visu_2 = capsule.addObject(
        "ConstantForceField",
        indices=[0],
        name="ceshi_2",
        forces=[[0, 0, 0, 0, 0, 0]],
        indexFromEnd=True,
        showArrowSize=info[5][0]["force_scale"][1],
    )

    capsule_visu1 = capsule.addChild("sphere_visu1")
    capsule_visu1.loader = capsule_visu1.addObject(
        "MeshSTLLoader",
        name="loader_visu1",
        filename="model/magnetic instrument/NS.stl",
        scale="0.0008",
        translation=[0.000, 0, 0],
        rotation=[0, 0, 0],
    )
    capsule_visu1.addObject(
        "OglModel",
        name="model_visu1",
        src="@loader_visu1",
        color=[1.0, 0.0, 0.0, 1.0],
        updateNormals=False,
    )
    capsule_visu1.addObject("RigidMapping")

    capsule_visu2 = capsule.addChild("sphere_visu2")
    capsule_visu2.loader = capsule_visu2.addObject(
        "MeshSTLLoader",
        name="loader_visu2",
        filename="model/magnetic instrument/NS.stl",
        scale="0.0008",
        translation=[-0.0028, 0, 0],
        rotation=[0, 0, 0],
    )
    capsule_visu2.addObject(
        "OglModel",
        name="model_visu2",
        src="@loader_visu2",
        color=[0.2824, 0.46275, 1.0, 1.0],
        updateNormals=False,
    )
    capsule_visu2.addObject("RigidMapping")

    for name, filename, color, translation, rotation in [
        ("sphere_visu3", "capsule_outer_shell.stl", [0.21176, 0.21176, 0.21176, 0.15], [0, 0, 0], [0, 0, 0]),
        ("sphere_visu4", "capsule_secondary_shell.stl", [0.21176, 0.21176, 0.21176, 0.15], [0, 0, 0], [0, 0, 0]),
        ("sphere_visu5", "capsule_body_insert.stl", [0.7098, 0.7098, 0.7098, 1], [0, 0, 0], [0, 0, 0]),
        ("sphere_visu6", "capsule_camera_module.stl", [0.5098, 0.5098, 0.5098, 1], [-0.006, 0, 0.0], [0, 0, 180]),
    ]:
        capsule_visu = capsule.addChild(name)
        capsule_visu.loader = capsule_visu.addObject(
            "MeshSTLLoader",
            name="loader_" + name,
            filename="model/magnetic instrument/" + filename,
            scale="0.0007",
            translation=translation,
            rotation=rotation,
        )
        capsule_visu.addObject(
            "OglModel",
            name="model_" + name,
            src="@loader_" + name,
            color=color,
            updateNormals=False,
        )
        capsule_visu.addObject("RigidMapping")

    collision = capsule.addChild("CubeCollisionModel1")
    collision.addObject(
        "MeshSTLLoader",
        name="loader",
        filename="model/magnetic instrument/capsule_collision_body.stl",
        triangulate=True,
        scale=0.0005,
        translation=[0, 0, 0],
        rotation=[0, 0, 90],
    )
    collision.addObject("MeshTopology", src="@loader")
    collision.addObject("MechanicalObject")
    collision.addObject("TriangleCollisionModel")
    collision.addObject("LineCollisionModel")
    collision.addObject("PointCollisionModel")
    collision.addObject("RigidMapping")
    return pose_capsule.position[0], pose_capsule, pose_capsule.velocity[0], force_torque_capsule, CFF_visu, CFF_visu_2


def Create_Manipulator(root_node, base_pose, magnet_pose, visual_type, X0=None):
    if visual_type != "mag":
        raise ValueError('Imitation-learned control requires magnetic-source visual_type="mag".')

    global _MANIPULATOR_COUNT
    robot_prefix = "robot" + str(_MANIPULATOR_COUNT)
    _MANIPULATOR_COUNT += 1

    if X0 is None:
        X0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    link_list = []
    link_pose_list = []
    link_visu_list = []
    a = np.array([[0], [-0.42500], [-0.39225], [0], [0], [0]])
    d = np.array([[0.089159], [0], [0], [0.10915], [0.09465], [0.08230]])
    alpha = np.array([[np.pi / 2], [0], [0], [np.pi / 2], [-np.pi / 2], [0]])
    T_bias = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.08089], [0, 0, 0, 1]])
    link_info = [
        {"filename": "model/magnetic source/base.stl", "translation": [-0.0, -0.0, 0.003], "rotation": [0, 90, 0], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J1.stl", "translation": [0, -0.0845, 0], "rotation": [90, 0, -90], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J2.stl", "translation": [-0.086, 0, 0.007], "rotation": [90, 0, 180], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J3.stl", "translation": [-0.034, -0.05, -0.062], "rotation": [90, 0, 0], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J4.stl", "translation": [0.002, 0, 0.002], "rotation": [0, -90, 0], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J5.stl", "translation": [0, 0.9982, -0.11], "rotation": [90, 0, 90], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/J6.stl", "translation": [0, 0.0, -0.035], "rotation": [90, 0, 0], "color": [0.7451, 0.7451, 0.7451, 1]},
        {"filename": "model/magnetic source/support.stl", "translation": [0, 0, -0.081], "rotation": [90, 0, -45], "color": [0.32157, 0.5451, 0.5451, 0.5]},
        {"filename": "model/magnetic source/north_pole.stl", "translation": [0, 0, 0], "rotation": [0, 0, 0], "color": [1, 0, 0, 1]},
        {"filename": "model/magnetic source/south_pole.stl", "translation": [0, 0, 0], "rotation": [0, 0, 0], "color": [0.2824, 0.46275, 1, 1]},
    ]

    robot = Manipulator2(base_pose, a, d, alpha, T_bias)
    position, moment = pose_to_position_moment(magnet_pose)
    theta = robot.back_numerical(position, moment, X0)
    link_init_pose_list = robot.fkine_all_link(theta)

    visu_object(
        root_node,
        path="model/magnetic source/box.stl",
        pose=[base_pose[0], base_pose[1], base_pose[2] - 0.725, 0.0, 0.7071067811865475, 0.0, -0.7071067811865475],
        scale="0.0009",
        color=[0.6509803, 0.6509803, 0.6509803, 1],
    )

    for i in range(len(link_init_pose_list)):
        link_list.append(root_node.addChild(robot_prefix + "_" + str(i)))
        link_pose_list.append(
            link_list[i].addObject(
                "MechanicalObject",
                name=robot_prefix + "_" + str(i),
                template="Rigid3",
                position=link_init_pose_list[i],
                showObject=1,
                showObjectScale=0.0,
            )
        )
        link_visu_list.append(link_list[i].addChild("robot_" + str(i) + "_visu"))
        link_visu_list[i].addObject(
            "MeshSTLLoader",
            name="loader" + str(i),
            filename=link_info[i]["filename"],
            translation=link_info[i]["translation"],
            rotation=link_info[i]["rotation"],
            scale3d=[0.001] * 3,
        )
        link_visu_list[i].addObject(
            "OglModel",
            name="model" + str(i),
            src="@loader" + str(i),
            color=link_info[i]["color"],
            updateNormals=False,
        )
        link_visu_list[i].addObject("RigidMapping")

    end_link = link_list[-1]
    for offset, info_idx in enumerate((8, 9), start=len(link_init_pose_list)):
        child = end_link.addChild("robot_" + str(offset) + "_visu")
        child.addObject(
            "MeshSTLLoader",
            name="loader" + str(offset),
            filename=link_info[info_idx]["filename"],
            translation=link_info[info_idx]["translation"],
            rotation=link_info[info_idx]["rotation"],
            scale3d=[0.001] * 3,
        )
        child.addObject(
            "OglModel",
            name="model" + str(offset),
            src="@loader" + str(offset),
            color=link_info[info_idx]["color"],
            updateNormals=False,
        )
        child.addObject("RigidMapping")

    return robot, theta, link_pose_list
