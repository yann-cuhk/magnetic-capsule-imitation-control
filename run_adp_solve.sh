#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "${PROJECT_DIR}/demo_c"
exec python3 solve_adp_capsule.py
