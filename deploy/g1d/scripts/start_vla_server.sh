#!/usr/bin/env bash
# RTX 5090 host (Ubuntu 24.04): serve NavGPT3 VLA for the robot.
#   CHECKPOINT=models/NavGPT3-8B GPU=0 PORT=8100 deploy/g1d/scripts/start_vla_server.sh
set -euo pipefail
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$WS"

CHECKPOINT="${CHECKPOINT:-models/NavGPT3-8B}"
GPU="${GPU:-0}"
PORT="${PORT:-8100}"
PY="${PY:-code/NavGPT-3/.envs/vla/bin/python}"

exec "$PY" deploy/g1d/server/serve_vla.py --checkpoint "$CHECKPOINT" --gpu "$GPU" --port "$PORT" "$@"
