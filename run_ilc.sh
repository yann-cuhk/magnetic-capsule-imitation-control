#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

mode="${1:-gui}"

if [[ "$mode" == "gui" ]]; then
  exec "${PROJECT_DIR}/run_gui.sh" "GUI_interface/info_magnetic_capsule_ilc_global.txt"
elif [[ "$mode" == "batch" || "$mode" == "online" ]]; then
  exec "${PROJECT_DIR}/run_batch.sh" "GUI_interface/info_magnetic_capsule_ilc_global.txt"
elif [[ "$mode" == "-h" || "$mode" == "--help" || "$mode" == "help" ]]; then
  cat <<'EOF'
Usage: ./run_ilc.sh [gui|batch]

Modes:
  gui      run Imitation-learned control with SOFA ImGui
  batch    run Imitation-learned control in headless batch mode
EOF
else
  echo "Unknown mode: ${mode}" >&2
  echo "Use './run_ilc.sh --help' for supported modes." >&2
  exit 2
fi
