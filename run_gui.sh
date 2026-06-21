#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFA_ROOT="${SOFA_ROOT:-${HOME}/Documents/SOFA}"
PY312_SITE="${PY312_SITE:-${HOME}/sofa_py312/lib/python3.12/site-packages}"

export SOFA_ROOT

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/magrobot-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ $# -gt 0 ]]; then
  export MAGROBOT_CONFIG="$1"
else
  export MAGROBOT_CONFIG="GUI_interface/info_magnetic_capsule_ilc_global.txt"
fi

export PYTHONPATH="${PY312_SITE}:${PROJECT_DIR}/source:${PROJECT_DIR}/source/Manipulator_Kinematics:${PROJECT_DIR}/source/Control_Package:${PROJECT_DIR}/source/Pose_Transform:${PROJECT_DIR}/source/Magnetic_Engine:${PROJECT_DIR}/source/Environment_Package:${PROJECT_DIR}/source/Module_Package"

cd "${PROJECT_DIR}/source"

exec "${SOFA_ROOT}/bin/runSofa" -g imgui -a -i \
  --noautoload \
  -l "${SOFA_ROOT}/plugins/SofaImGui/lib/libSofaImGui.dylib" \
  -l "${SOFA_ROOT}/plugins/SofaPython3/lib/libSofaPython3.dylib" \
  -l "${SOFA_ROOT}/plugins/BeamAdapter/lib/libBeamAdapter.dylib" \
  -l "${SOFA_ROOT}/plugins/SoftRobots/lib/libSoftRobots.dylib" \
  -l "${SOFA_ROOT}/plugins/ArticulatedSystemPlugin/lib/libArticulatedSystemPlugin.dylib" \
  main.py
