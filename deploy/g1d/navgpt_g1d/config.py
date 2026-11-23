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
    odom: str = "rt/slamware_ros_sdk_server_node/odom"   # with g1d_ws running: rt/nav/base_odom (LIO + wheels)
    scan: str = "rt/slamware_ros_sdk_server_node/scan"
    cmd_vel: str = "rt/cmd_vel_no_limit"                  # written in direct mode
    guard_cmd: str = "rt/nav/cmd_vel"                     # written in guard mode
    guard_state: str = "rt/nav/guard_state"
    joystick: str = "rt/wirelesscontroller"


@dataclass(frozen=True)
class BaseConfig:
    output: str = "direct"                # direct: write cmd_vel; guard: hand commands to the base guard
    other_writers: str = "refuse"         # direct mode: refuse | warn when cmd_vel already has a writer
    publish_hz: float = 30.0
    cmd_timeout_s: float = 0.3
    zero_hold_s: float = 1.0              # direct mode; the guard does its own zero hold
    guard_release_s: float = 0.2
    cmd_x_sign: float = 1.0               # direct mode only; in guard mode the guard owns the base
    cmd_yaw_sign: float = 1.0
    odom_yaw_sign: float = 1.0


@dataclass(frozen=True)
class CameraConfig:
    host: str = "192.168.123.164"
    port: Optional[int] = None            # null: ask teleimager (request_port) for the head camera's port
    binocular: Optional[bool] = None      # null: from the teleimager config
    request_port: int = 60000
    name: str = "head_camera"
    eye: str = "left"
    out_size: int = 400
    stale_after_s: float = 0.3
    latency_s: float = 0.06               # capture-to-receipt delay


@dataclass(frozen=True)
class SupervisorConfig:
    joystick: bool = True                 # direct mode: remote sticks end the episode
    joystick_deadzone: float = 0.15
    chassis_http: Optional[str] = "http://192.168.123.163:1448"   # direct mode bumper/cliff events; null disables
    estop_events: tuple = ("BUMPER_TRIGGERED", "CLIFF_DETECTED")
    event_poll_s: float = 0.2
    guard_state_timeout_s: float = 0.5    # guard mode: older guard state ends the episode
    guard_block_timeout_s: float = 20.0   # guard mode: held back this long ends the episode as blocked


@dataclass(frozen=True)
class RecordConfig:
    rate_hz: float = 4.0                  # dense frames for navgpt_sft (keyframes are picked at training time)
    out_dir: str = "data/sft/g1d_teleop"
    manifest: str = "train.jsonl"
    jpeg_quality: int = 95


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
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
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
               "safety": SafetyParams, "supervisor": SupervisorConfig, "episode": EpisodeConfig,
               "record": RecordConfig, "api": ApiConfig}
    built = {name: _build(cls, raw.get(name) or {}, name) for name, cls in classes.items()}
    built["navigator"] = dataclasses.replace(built["navigator"], tracker=built["tracker"])
    built["supervisor"] = dataclasses.replace(built["supervisor"], estop_events=tuple(built["supervisor"].estop_events))
    cfg = AppConfig(**built)
    _check(cfg)
    return cfg


def _check(cfg: AppConfig) -> None:
    if cfg.base.output not in ("direct", "guard"):
        raise ValueError("base.output must be direct or guard, got {!r}".format(cfg.base.output))
    if cfg.base.other_writers not in ("refuse", "warn"):
        raise ValueError("base.other_writers must be refuse or warn, got {!r}".format(cfg.base.other_writers))
    if cfg.camera.eye not in ("left", "right"):
        raise ValueError("camera.eye must be left or right, got {!r}".format(cfg.camera.eye))
