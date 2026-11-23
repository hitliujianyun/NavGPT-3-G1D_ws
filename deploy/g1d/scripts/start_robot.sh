#!/usr/bin/env bash
# Robot side (PC2): start the control API, then POST /episode to run instructions.
#   CONFIG=config/g1d.yaml deploy/g1d/scripts/start_robot.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/g1d.yaml}"
PY="${PY:-python3}"

exec "$PY" -m navgpt_g1d serve --config "$CONFIG" "$@"
