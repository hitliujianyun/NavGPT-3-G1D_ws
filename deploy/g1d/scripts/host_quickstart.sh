#!/usr/bin/env bash
# Day-1 bring-up from the RTX 5090 host (Ubuntu 24.04) over a network cable to the robot LAN.
# Everything runs on the host (VLA server + navgpt_g1d); PC2 is not touched.
#
#   deploy/g1d/scripts/host_quickstart.sh <nic-connected-to-the-robot> [--set-ip]
#
# Creates deploy/g1d/.venv-host with CycloneDDS (PyPI wheel; tested with unitree_sdk2py at 11.1.0),
# unitree_sdk2py (from the submodule, --no-deps because it pins cyclonedds 0.10.2), pyzmq, OpenCV.
set -euo pipefail
NIC="${1:?usage: host_quickstart.sh <nic connected to the robot> [--set-ip]}"
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
G1D="$WS/deploy/g1d"
VENV="$G1D/.venv-host"
IP=192.168.123.222

if [[ "${2:-}" == "--set-ip" ]]; then
  sudo nmcli con add type ethernet ifname "$NIC" con-name g1d ipv4.method manual ipv4.addresses "$IP/24" 2>/dev/null || true
  sudo nmcli con up g1d
fi
if ip -br addr show "$NIC" | grep -q "$IP/"; then echo "$NIC has $IP"; else echo "WARN: $NIC has no $IP (rerun with --set-ip)"; fi
if ping -c 2 -W 1 192.168.123.164 >/dev/null; then echo "PC2 192.168.123.164 reachable"; else echo "WARN: PC2 not reachable"; fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "cyclonedds==11.1.0" pyzmq opencv-python-headless pyyaml numpy requests
"$VENV/bin/pip" install -q --no-deps -e "$WS/code/unitree_sdk2_python"
"$VENV/bin/pip" install -q --no-deps -e "$G1D"
"$VENV/bin/python" -c "import cyclonedds, unitree_sdk2py, zmq, cv2; print('robot-side environment ok')"

if command -v ufw >/dev/null && sudo ufw status | grep -q "Status: active"; then
  echo "WARN: ufw is active; allow DDS from the robot LAN: sudo ufw allow from 192.168.123.0/24"
fi

cat <<EOF

next (VLA server in another terminal: CHECKPOINT=models/NavGPT3-8B GPU=0 deploy/g1d/scripts/start_vla_server.sh):
  cd $G1D
  $VENV/bin/python -m navgpt_g1d check --config config/host_wired.yaml
  $VENV/bin/python -m navgpt_g1d smoke --config config/host_wired.yaml
  $VENV/bin/python -m navgpt_g1d run   --config config/host_wired.yaml --dry-run --instruction "Walk forward and stop at the door."
EOF
