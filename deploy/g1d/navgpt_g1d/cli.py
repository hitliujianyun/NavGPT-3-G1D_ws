"""Command line: python -m navgpt_g1d <command> ...

    run     one VLN episode on the robot
    serve   robot control API (start/stop episodes over HTTP)
    sim     one episode against the simulated base and an oracle VLA
    check   sensor rates, camera, guard / remote / chassis events, and VLA reachability
    smoke   short motions to verify signs against odometry
    record  teleoperated demonstrations for navgpt_sft (the remote drives)
    report  smoothness and latency metrics of a logged episode
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import signal
import sys
import threading
import time

from .config import load_config
from .geometry import wrap_angle, world_to_body


def _sigterm_as_interrupt() -> None:
    """systemd stops services with SIGTERM: take the same safe path as Ctrl-C."""
    def handler(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:      # not the main thread
        pass


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _robot(cfg, dry_run: bool, with_camera: bool = True):
    from .robot import G1DRobot, PreflightError

    try:
        return G1DRobot(cfg, dry_run=dry_run, with_camera=with_camera)
    except PreflightError as exc:
        raise SystemExit("[navgpt_g1d] preflight failed: {}".format(exc))


def _robot_and_vla(cfg, dry_run: bool):
    from .vla_client import VlaClient

    return _robot(cfg, dry_run), VlaClient(cfg.vla.url, cfg.vla.timeout_s, cfg.vla.jpeg_quality)


def cmd_run(args) -> int:
    from .episode import EpisodeRunner
    from .report import analyse

    _sigterm_as_interrupt()
    cfg = load_config(args.config)
    robot, vla = _robot_and_vla(cfg, args.dry_run)
    try:
        print("[navgpt_g1d] VLA:", vla.info())
        runner = EpisodeRunner(cfg, robot, vla, detect_stuck=not args.dry_run)
        try:
            result = runner.run(args.instruction)
        except KeyboardInterrupt:
            runner.stop("operator interrupt")
            result = runner.wait()
        _print(dataclasses.asdict(result))
        _print(analyse(result.log_dir))
        return 0 if result.status == "arrived" else 1
    finally:
        robot.close()


def cmd_serve(args) -> int:
    from .api import serve
    from .episode import EpisodeRunner

    _sigterm_as_interrupt()
    cfg = load_config(args.config)
    if args.sim:
        from .sim import ROUTES, RouteOracleVLA, SimRobot, SimWorld

        world = SimWorld(odom_noise_m=0.005)
        robot = SimRobot(world, cfg.base.publish_hz, cfg.base.cmd_timeout_s)
        vla = RouteOracleVLA(world, ROUTES[args.route])
    else:
        robot, vla = _robot_and_vla(cfg, args.dry_run)
    try:
        serve(EpisodeRunner(cfg, robot, vla, detect_stuck=not args.dry_run), robot, vla, cfg.api.host, cfg.api.port)
    finally:
        robot.close()
    return 0


def cmd_sim(args) -> int:
    from .episode import EpisodeRunner
    from .report import analyse
    from .sim import ROUTES, Obstacle, RouteOracleVLA, SimRobot, SimWorld

    cfg = load_config(args.config)
    cfg = dataclasses.replace(cfg, episode=dataclasses.replace(cfg.episode, save_frames=False,
                                                               log_dir=args.log_dir or cfg.episode.log_dir))
    route = ROUTES[args.route]
    obstacles = []
    if args.crossing:
        x = route[1][0] * 0.5
        obstacles.append(Obstacle(x=x, y=-2.0, vy=0.6, radius=0.25, t_on=args.crossing_at))
    world = SimWorld(obstacles=obstacles, odom_noise_m=0.005, seed=args.seed)
    robot = SimRobot(world, cfg.base.publish_hz, cfg.base.cmd_timeout_s)
    vla = RouteOracleVLA(world, route, seed=args.seed)
    try:
        result = EpisodeRunner(cfg, robot, vla).run("simulated route " + args.route)
        _print(dataclasses.asdict(result))
        _print(analyse(result.log_dir))
        return 0 if result.status == "arrived" else 1
    finally:
        robot.close()


def cmd_check(args) -> int:
    from .dds import find_writers

    cfg = load_config(args.config)
    robot, vla = _robot_and_vla(cfg, dry_run=True)
    try:
        print("output mode: {}; camera: {}".format(cfg.base.output, robot.camera_info))
        try:
            writers = find_writers(cfg.topics.cmd_vel, cfg.dds.domain_id, 1.5)
            print("writers on {}: {}".format(cfg.topics.cmd_vel, len(writers)),
                  "(expected 0 in direct mode, 1 = the guard in guard mode)")
        except Exception as exc:  # noqa: BLE001
            print("writer discovery unavailable: {!r}".format(exc))
        t0 = time.monotonic()
        try:
            info = vla.info()
            print("VLA {} reachable in {:.0f} ms: {}".format(cfg.vla.url, 1000 * (time.monotonic() - t0), info))
        except Exception as exc:  # noqa: BLE001
            print("VLA {} NOT reachable: {!r}".format(cfg.vla.url, exc))
        for _ in range(int(args.seconds / 0.5)):
            time.sleep(0.5)
            health = robot.health()
            frame = robot.frame()
            health["frame"] = None if frame is None else list(frame[0].shape)
            scan = robot.scan()
            health["scan_points"] = None if scan is None else int(len(scan.ranges))
            print(json.dumps(health, default=str))
        return 0
    finally:
        robot.close()


def cmd_smoke(args) -> int:
    """Drive forward and turn left briefly; compare with odometry to verify signs."""
    cfg = load_config(args.config)
    robot = _robot(cfg, dry_run=False, with_camera=False)
    try:
        deadline = time.monotonic() + 3.0
        while robot.odom.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if robot.odom.latest() is None:
            print("no odometry on", cfg.topics.odom)
            return 1
        steps = [("forward", args.speed, 0.0), ("turn left (counter-clockwise seen from above)", 0.0, args.turn_rate)]
        for label, v, w in steps:
            input("Clear the area, then press Enter to {} for {:.1f} s ".format(label, args.seconds))
            p0 = robot.odom.latest()[1]
            robot.cmd.activate()
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                robot.cmd.set(v, w)
                time.sleep(0.05)
            robot.cmd.release(cfg.base.zero_hold_s)
            time.sleep(0.5)
            p1 = robot.odom.latest()[1]
            dx, dy = world_to_body(p0, p1[0], p1[1])
            dyaw = wrap_angle(p1[2] - p0[2])
            print("odometry change: forward {:+.3f} m, left {:+.3f} m, yaw {:+.1f} deg".format(dx, dy, math.degrees(dyaw)))
            seen = input("Which way did the robot actually move? [f]orward/[b]ack/[l]eft turn/[r]ight turn/[n]one: ")
            _advise(label, seen.strip().lower()[:1], dx, dyaw)
        return 0
    finally:
        robot.close()


def _advise(label: str, seen: str, dx: float, dyaw: float) -> None:
    if label == "forward":
        if seen == "b":
            print("-> robot moved backwards: set base.cmd_x_sign: -1.0")
        elif seen == "f" and dx < -0.03:
            print("-> robot moved forward but odometry says backwards: check the odometry frame")
        elif seen == "f":
            print("-> forward OK")
        else:
            print("-> no motion: check that /cmd_vel_no_limit reaches the base (remote in the right mode?)")
    else:
        if seen == "r":
            print("-> robot turned right: set base.cmd_yaw_sign: -1.0")
        elif seen == "l" and dyaw < -0.05:
            print("-> robot turned left but odometry yaw decreased: set base.odom_yaw_sign: -1.0")
        elif seen == "l":
            print("-> turn OK")
        else:
            print("-> no rotation: check the base mode")


def cmd_record(args) -> int:
    """Interactive: one episode per instruction; the remote drives, nothing is commanded."""
    from .record import EpisodeRecorder

    _sigterm_as_interrupt()
    cfg = load_config(args.config)
    rec_cfg = cfg.record
    out_dir = args.out or rec_cfg.out_dir
    robot = _robot(cfg, dry_run=True)
    try:
        deadline = time.monotonic() + 3.0
        while (robot.odom.latest() is None or robot.frame() is None) and time.monotonic() < deadline:
            time.sleep(0.05)
        if robot.odom.latest() is None or robot.frame() is None:
            print("need odometry ({}) and camera frames ({})".format(cfg.topics.odom, robot.camera_info))
            return 1
        print("recording to {}/{} at {:.1f} Hz; drive with the remote".format(out_dir, rec_cfg.manifest, rec_cfg.rate_hz))
        kept = 0
        while True:
            text = args.instruction or input("\ninstruction (English; several separated by '|'; empty = quit): ")
            if not text.strip():
                break
            if args.duration is None:
                input("put the robot at the start, press Enter to record...")
            rec = EpisodeRecorder(robot, out_dir, cfg.camera.latency_s, rec_cfg.rate_hz, rec_cfg.jpeg_quality)
            stop = threading.Event()
            if args.duration is None:
                threading.Thread(target=lambda: (input("recording... press Enter at the goal\n"), stop.set()),
                                 daemon=True).start()
                rec.run(stop.is_set)
            else:
                rec.run(stop.is_set, max_s=args.duration)
            print("{} frames, {:.2f} m, {} skipped".format(len(rec.frames), rec.path_length(), rec.skipped))
            keep = args.instruction is not None or input("keep this episode? [Y/n] ").strip().lower() != "n"
            line = rec.finish(text.split("|"), rec_cfg.manifest, rate_hz=rec_cfg.rate_hz) if keep else None
            if line is None:
                shutil.rmtree(rec.dir, ignore_errors=True)
                print("discarded")
            else:
                kept += 1
                print("saved {}".format(line["episode_id"]))
            if args.instruction:
                break
        print("{} episode(s) kept in {}/{}".format(kept, out_dir, rec_cfg.manifest))
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        robot.close()


def cmd_report(args) -> int:
    from .report import analyse

    _print(analyse(args.log_dir))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m navgpt_g1d", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="one VLN episode on the robot")
    p.add_argument("--config", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--dry-run", action="store_true", help="everything live except /cmd_vel_no_limit")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("serve", help="robot control API")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sim", action="store_true", help="serve the simulated base (UI/planner integration)")
    p.add_argument("--route", choices=["straight", "L", "S"], default="L", help="route for --sim")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("sim", help="simulated base + oracle VLA")
    p.add_argument("--config", default=None)
    p.add_argument("--route", choices=["straight", "L", "S"], default="L")
    p.add_argument("--crossing", action="store_true", help="a person crosses the route")
    p.add_argument("--crossing-at", type=float, default=3.0, help="seconds after start")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default=None)
    p.set_defaults(fn=cmd_sim)

    p = sub.add_parser("check", help="sensor rates, camera, VLA")
    p.add_argument("--config", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("smoke", help="verify cmd_vel / odometry signs")
    p.add_argument("--config", required=True)
    p.add_argument("--speed", type=float, default=0.15)
    p.add_argument("--turn-rate", type=float, default=0.3)
    p.add_argument("--seconds", type=float, default=1.5)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("record", help="teleoperated demonstrations for navgpt_sft")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="dataset directory (default: record.out_dir)")
    p.add_argument("--instruction", default=None, help="record a single episode with this instruction")
    p.add_argument("--duration", type=float, default=None, help="seconds per episode instead of Enter")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("report", help="metrics of a logged episode")
    p.add_argument("log_dir")
    p.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
