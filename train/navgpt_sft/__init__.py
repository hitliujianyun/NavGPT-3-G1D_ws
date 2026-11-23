"""LoRA + action-head fine-tuning of NavGPT VLA (NavGPT3) with a frozen vision tower."""

import sys
from pathlib import Path

try:
    import navgpt  # noqa: F401
except ImportError:  # outside the VLA environment's editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code" / "NavGPT-3"))
