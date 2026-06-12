#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
exec "${PROJECT_DIR}/run_batch.sh" "GUI_interface/info_case3_single_capsule_stomach_adp_collect.txt"
