# MagRobot Mac Deployment

This folder is a macOS deployment copy of MagRobot for the local SOFA install at:

`/Users/young/Documents/SOFA`

The original Windows bundle remains unchanged at:

`/Users/young/Documents/MagRobot`

## Run

Validate with SOFA batch mode:

```bash
/Users/young/Documents/MagRobot_mac/run_batch.sh
```

Open the SOFA GUI:

```bash
/Users/young/Documents/MagRobot_mac/run_gui.sh
```

Run a different MagRobot config:

```bash
/Users/young/Documents/MagRobot_mac/run_gui.sh GUI_interface/info_wire_elc_soft_arc3_nc.txt
```

## Notes

- The Windows `MNSS.exe` and `.dll` files are not used on macOS.
- Python dependencies are installed in `/Users/young/sofa_py312`.
- The deployment copy includes compatibility updates for SOFA 25.12 and NumPy 2.
- The default scene is `demo_c/GUI_interface/ceshi.txt`.
