# single_capsule_stomach

This is a focused MagRobot macOS package for the single-manipulator, single-capsule, soft-stomach scene.

## Scene

- Config: `demo_c/GUI_interface/info_case3_single_capsule_stomach.txt`
- Environment: `soft_static` stomach using `model/environment/stomach.stl` and `model/environment/stomach.msh`
- Instrument: one magnetic capsule
- Magnetic source: one robotic manipulator carrying a permanent magnet
- Path mode: `yushe`, type `stomach_small`
- Control mode: automatic `PID`

This scene is automatic. Keyboard manual controls are not the main control path for this config.

## Run

From this folder:

```bash
./run_gui.sh
```

Batch validation:

```bash
./run_batch.sh
```

The scripts expect the local SOFA install at `/Users/young/Documents/SOFA` and Python dependencies in `/Users/young/sofa_py312`, matching the existing Mac deployment.

## Contents

- `demo_c/main.py`: SOFA scene entry point
- `demo_c/create_scene.py`: scene assembly
- `demo_c/simulator_setting.py`: SOFA physics setup
- `demo_c/Module_Package`: SOFA object and controller helpers
- `demo_c/Magnetic_Engine`: magnetic field, force, torque, and inverse calculations
- `demo_c/model`: required mesh and visualization assets
- `demo_c/GUI_interface/info_case3_single_capsule_stomach.txt`: the selected scene config

Large test reports from the full repository are intentionally excluded.
