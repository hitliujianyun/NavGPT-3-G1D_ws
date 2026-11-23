"""One YAML file configures the robot side; unknown keys are rejected."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .navigator import NavigatorParams
from .safety import SafetyParams
from .tracker import TrackerParams


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int = 0
    interface: Optional[str] = None       # NIC name, e.g. eth0
    interface_ip: Optional[str] = None    # or the NIC's address on 192.168.123.x (preferred)


@dataclass(frozen=True)
class TopicsConfig:
    odom: str = "rt/slamware_ros_sdk_server_node/odom"
    scan: str = "rt/slamware_ros_sdk_server_node/scan"
    cmd_vel: str = "rt/cmd_vel_no_limit"


@dataclass(frozen=True)
class BaseConfig:
    publish_hz: float = 30.0
    cmd_timeout_s: float = 0.3
    zero_hold_s: float = 1.0
    cmd_x_sign: float = 1.0
    cmd_yaw_sign: float = 1.0
    odom_yaw_sign: float = 1.0


@dataclass(frozen=True)
class CameraConfig:
    host: str = "192.168.123.164"
    port: int = 55555
    binocular: bool = True
    eye: str = "left"
    out_size: int = 400
    stale_after_s: float = 0.3
    latency_s: float = 0.06               # capture-to-receipt delay


@dataclass(frozen=True)
class VlaConfig:
    url: str = "http://127.0.0.1:8100"
    timeout_s: float = 4.0
    jpeg_quality: int = 85


@dataclass(frozen=True)
class EpisodeConfig:
    control_hz: float = 20.0
    max_time_s: float = 300.0
    max_requests: int = 2000
    odom_timeout_s: float = 0.5
    stuck_time_s: float = 4.0
    stuck_dist_m: float = 0.05
    min_request_period_s: float = 0.0     # 0 = back-to-back requests
    log_dir: str = "logs"
    save_frames: bool = True


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8124


@dataclass(frozen=True)
class AppConfig:
    dds: DdsConfig = field(default_factory=DdsConfig)
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vla: VlaConfig = field(default_factory=VlaConfig)
    tracker: TrackerParams = field(default_factory=TrackerParams)
    navigator: NavigatorParams = field(default_factory=NavigatorParams)
    safety: SafetyParams = field(default_factory=SafetyParams)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def _build(cls, data: dict, where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError("unknown key(s) {} in {}; valid: {}".format(sorted(unknown), where, sorted(names)))
    return cls(**data)


def load_config(path=None) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    raw = raw or {}
    sections = {f.name: f.type for f in dataclasses.fields(AppConfig)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError("unknown section(s) {}; valid: {}".format(sorted(unknown), sorted(sections)))
    classes = {"dds": DdsConfig, "topics": TopicsConfig, "base": BaseConfig, "camera": CameraConfig,
               "vla": VlaConfig, "tracker": TrackerParams, "navigator": NavigatorParams,
               "safety": SafetyParams, "episode": EpisodeConfig, "api": ApiConfig}
    built = {name: _build(cls, raw.get(name) or {}, name) for name, cls in classes.items()}
    built["navigator"] = dataclasses.replace(built["navigator"], tracker=built["tracker"])
    return AppConfig(**built)
