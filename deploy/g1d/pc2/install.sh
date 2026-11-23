#!/usr/bin/env bash
# Install navgpt_g1d on the G1-D PC2 (Jetson Orin NX, JetPack 5.1.1 / Ubuntu 20.04).
#
#   tar xzf navgpt_g1d-<commit>.tar.gz && sudo navgpt_g1d/pc2/install.sh [--enable] [--python PY] [--no-tests]
#
# Installs to /opt/navgpt_g1d (an existing install is kept as /opt/navgpt_g1d.old-<time>), keeps the
# site configuration in /opt/navgpt_g1d_site.yaml across upgrades, picks a Python that can import
# unitree_sdk2py + pyzmq + OpenCV (system python3 or the PC2 image's conda env "tv"), installs the
# systemd unit, and runs the self-tests.
set -euo pipefail

ROOT=/opt/navgpt_g1d
SITE=/opt/navgpt_g1d_site.yaml
RUN_USER=unitree
ENABLE=0
TESTS=1
PY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable) ENABLE=1 ;;
    --no-tests) TESTS=0 ;;
    --python) PY="$2"; shift ;;
    --user) RUN_USER="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
HERE="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$(getent passwd "$RUN_USER" | cut -d: -f6)"

# 1. Files
if [[ "$HERE" != "$ROOT" ]]; then
  if [[ -d "$ROOT" ]]; then
    systemctl stop navgpt-g1d 2>/dev/null || true
    mv "$ROOT" "$ROOT.old-$(date +%Y%m%d-%H%M%S)"
  fi
  cp -a "$HERE" "$ROOT"
fi
mkdir -p "$ROOT/logs"
chown -R "$RUN_USER:$RUN_USER" "$ROOT"

# 2. Python: needs unitree_sdk2py (+ cyclonedds), pyzmq, OpenCV, numpy, PyYAML, requests
PYPATH="$ROOT:$ROOT/third_party/unitree_sdk2_python"
as_user() { sudo -u "$RUN_USER" -H env PYTHONPATH="$PYPATH" "$@"; }    # sudo drops PYTHONPATH otherwise
probe() { as_user "$1" -c "import cyclonedds, unitree_sdk2py, zmq, cv2, numpy, yaml, requests" 2>/dev/null; }
CANDIDATES=("$PY" /usr/bin/python3)
for base in miniconda3 anaconda3 miniforge3 mambaforge; do
  CANDIDATES+=("$HOME_DIR/$base/envs/tv/bin/python")
done
FOUND=""
for c in "${CANDIDATES[@]}"; do
  [[ -n "$c" && -x "$c" ]] || continue
  if probe "$c"; then FOUND="$c"; break; fi
done
if [[ -z "$FOUND" ]]; then
  PYBASE="${PY:-/usr/bin/python3}"
  echo "No interpreter has every dependency; installing into $PYBASE (user site) ..."
  sudo -u "$RUN_USER" -H "$PYBASE" -m pip install --user "pyzmq>=19" pyyaml requests numpy || true
  if ! sudo -u "$RUN_USER" -H "$PYBASE" -c "import cyclonedds" 2>/dev/null; then
    echo "  cyclonedds missing: build CycloneDDS 0.10.x, export CYCLONEDDS_HOME, then"
    echo "  pip3 install -e $ROOT/third_party/unitree_sdk2_python   (see its README)"
  fi
  sudo -u "$RUN_USER" -H "$PYBASE" -c "import cv2" 2>/dev/null || echo "  OpenCV missing: sudo apt install python3-opencv"
  probe "$PYBASE" || { echo "dependencies still missing; fix the messages above and rerun"; exit 1; }
  FOUND="$PYBASE"
fi
echo "python: $FOUND ($("$FOUND" --version 2>&1))"
sed -i "s|^NAVGPT_G1D_PYTHON=.*|NAVGPT_G1D_PYTHON=$FOUND|" "$ROOT/pc2/navgpt_g1d.env"

# 3. Site configuration (kept across upgrades)
if [[ ! -f "$SITE" ]]; then
  cp "$ROOT/config/g1d.yaml" "$SITE"
  chown "$RUN_USER:$RUN_USER" "$SITE"
  echo "created $SITE from config/g1d.yaml: fill in the 'verify' items (vla.url, dds.interface_ip, ...)"
fi

# 4. systemd
install -m 644 "$ROOT/pc2/navgpt-g1d.service" /etc/systemd/system/navgpt-g1d.service
sed -i "s|^User=.*|User=$RUN_USER|" /etc/systemd/system/navgpt-g1d.service
systemctl daemon-reload
if [[ $ENABLE -eq 1 ]]; then systemctl enable navgpt-g1d; fi

# 5. Self-tests (simulation, no robot motion)
if [[ $TESTS -eq 1 ]]; then
  (cd "$ROOT" && as_user "$FOUND" -m unittest discover -s tests) || { echo "self-tests failed"; exit 1; }
fi

cat <<EOF

installed to $ROOT (site config $SITE)
next:
  set -a; source $ROOT/pc2/navgpt_g1d.env; set +a; cd $ROOT
  \$NAVGPT_G1D_PYTHON -m navgpt_g1d check --config \$NAVGPT_G1D_CONFIG
  \$NAVGPT_G1D_PYTHON -m navgpt_g1d smoke --config \$NAVGPT_G1D_CONFIG      # direct mode only, robot moves
  sudo systemctl start navgpt-g1d && journalctl -fu navgpt-g1d
EOF
