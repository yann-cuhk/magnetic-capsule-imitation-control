import json
import os
import simulator_setting
import create_scene
import animate_controller
import path_controller
import position_estimate

def createScene(root_node):
    config_path = os.environ.get("MAGROBOT_CONFIG", "GUI_interface/info_magnetic_capsule_ilc_global.txt")
    f = open(config_path, "r", encoding="UTF-8")
    json_str = f.read()
    initial_info = json.loads(json_str)
    simulator = simulator_setting.Simulator(root_node, info=initial_info)
    scene = create_scene.Scene(root_node, info=initial_info)
    path = None
    pose_estimate = None
    if initial_info[6] and initial_info[2]:
        path = path_controller.Path(root_node, initial_info, scene)
    if initial_info[7] and initial_info[2]:
        pose_estimate = position_estimate.Position_Estimate(initial_info, scene, name='pose_estimate_controller')
        root_node.addObject(pose_estimate)
    if path and pose_estimate:
        mag_controller = animate_controller.MagController(root_node, info=initial_info, simulator=simulator, scene=scene, path=path,
                                                          pose_estimate=pose_estimate, name='mag_controller')
        root_node.addObject(mag_controller)
