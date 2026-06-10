#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFA_ROOT="/Users/young/Documents/SOFA"
PY312_SITE="/Users/young/sofa_py312/lib/python3.12/site-packages"

export SOFA_ROOT
export MPLCONFIGDIR="/private/tmp/magrobot-matplotlib"
export MAGROBOT_COMMAND_FILE="${MAGROBOT_COMMAND_FILE:-/private/tmp/magrobot_manual_key.json}"
mkdir -p "$MPLCONFIGDIR"
rm -f "$MAGROBOT_COMMAND_FILE"

if [[ $# -gt 0 ]]; then
  export MAGROBOT_CONFIG="$1"
else
  export MAGROBOT_CONFIG="GUI_interface/info_case3_single_capsule_stomach.txt"
fi

export PYTHONPATH="${PY312_SITE}:${PROJECT_DIR}/demo_c:${PROJECT_DIR}/demo_c/Manipulator_Kinematics:${PROJECT_DIR}/demo_c/Control_Package:${PROJECT_DIR}/demo_c/Pose_Transform:${PROJECT_DIR}/demo_c/State_Estimation:${PROJECT_DIR}/demo_c/Trajectory_Package:${PROJECT_DIR}/demo_c/Magnetic_Engine:${PROJECT_DIR}/demo_c/Environment_Package:${PROJECT_DIR}/demo_c/Module_Package"

cd "${PROJECT_DIR}/demo_c"
exec "${SOFA_ROOT}/bin/runSofa" -g imgui -a -i \
  --noautoload \
  -l "${SOFA_ROOT}/plugins/SofaImGui/lib/libSofaImGui.dylib" \
  -l "${SOFA_ROOT}/plugins/SofaPython3/lib/libSofaPython3.dylib" \
  -l "${SOFA_ROOT}/plugins/BeamAdapter/lib/libBeamAdapter.dylib" \
  -l "${SOFA_ROOT}/plugins/SoftRobots/lib/libSoftRobots.dylib" \
  -l "${SOFA_ROOT}/plugins/ArticulatedSystemPlugin/lib/libArticulatedSystemPlugin.dylib" \
  main.py
