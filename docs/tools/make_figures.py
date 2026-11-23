"""Figures for docs/并行推理与平滑轨迹_功能详解.html, drawn from the deployment code itself.

    python docs/tools/make_figures.py

Writes SVG files to docs/assets/ (no plotting library needed):
    fig_splice.svg      old path, new prediction, position blend vs heading blend
    fig_trajectory.svg  overlapped run vs stop-and-go on the L route
    fig_speed.svg       commanded speed over time, overlapped vs stop-and-go
    fig_labels.svg      train/navgpt_sft: keyframes and the eight-waypoint target of a dense episode
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "deploy" / "g1d"), str(ROOT / "deploy" / "g1d" / "tests"), str(ROOT / "train")]

import test_navigator_sim as nsim  # noqa: E402
from navgpt_g1d.geometry import integrate_unicycle  # noqa: E402
from navgpt_g1d.navigator import suggests_arrival  # noqa: E402
from navgpt_g1d.path import _from_points, _smoothstep, resample, splice  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
COLORS = {"route": "#94a3b8", "old": "#64748b", "new": "#4f46e5", "pos": "#dc2626", "head": "#0f766e",
          "sg": "#d97706", "robot": "#111827"}


class Plot:
    def __init__(self, w, h, xlim, ylim, title, xlabel, ylabel, equal=False, margin=(58, 20, 42, 46)):
        self.w, self.h = w, h
        self.ml, self.mr, self.mt, self.mb = margin
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        pw, ph = w - self.ml - self.mr, h - self.mt - self.mb
        if equal:
            s = min(pw / (self.x1 - self.x0), ph / (self.y1 - self.y0))
            self.sx = self.sy = s
        else:
            self.sx, self.sy = pw / (self.x1 - self.x0), ph / (self.y1 - self.y0)
        self.parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" font-family="Segoe UI, Microsoft YaHei, sans-serif">'.format(w, h),
                      '<rect width="100%" height="100%" fill="#ffffff"/>',
                      '<text x="{}" y="22" font-size="15" font-weight="600" fill="#1d2330">{}</text>'.format(self.ml, title)]
        self._axes(xlabel, ylabel)
        self.legend_items = []

    def X(self, x):
        return self.ml + (x - self.x0) * self.sx

    def Y(self, y):
        return self.mt + (self.y1 - y) * self.sy

    def _ticks(self, lo, hi):
        span = hi - lo
        step = 10 ** math.floor(math.log10(span / 5))
        for m in (1, 2, 5, 10):
            if span / (step * m) <= 7:
                step *= m
                break
        start = math.ceil(lo / step) * step
        return [round(start + i * step, 6) for i in range(int((hi - start) / step) + 1)]

    def _axes(self, xlabel, ylabel):
        x_end, y_end = self.X(self.x1), self.Y(self.y0)
        for t in self._ticks(self.x0, self.x1):
            x = self.X(t)
            self.parts.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2:.1f}" stroke="#eef0f4"/>'.format(x, self.mt, y_end))
            self.parts.append('<text x="{:.1f}" y="{:.1f}" font-size="11" fill="#5b6475" text-anchor="middle">{:g}</text>'.format(x, y_end + 15, t))
        for t in self._ticks(self.y0, self.y1):
            y = self.Y(t)
            self.parts.append('<line x1="{0}" y1="{1:.1f}" x2="{2:.1f}" y2="{1:.1f}" stroke="#eef0f4"/>'.format(self.ml, y, x_end))
            self.parts.append('<text x="{}" y="{:.1f}" font-size="11" fill="#5b6475" text-anchor="end">{:g}</text>'.format(self.ml - 6, y + 4, t))
        self.parts.append('<rect x="{}" y="{}" width="{:.1f}" height="{:.1f}" fill="none" stroke="#cbd1dc"/>'.format(
            self.ml, self.mt, x_end - self.ml, y_end - self.mt))
        self.parts.append('<text x="{:.1f}" y="{}" font-size="12" fill="#1d2330" text-anchor="middle">{}</text>'.format(
            (self.ml + x_end) / 2, self.h - 8, xlabel))
        self.parts.append('<text x="14" y="{:.1f}" font-size="12" fill="#1d2330" text-anchor="middle" transform="rotate(-90 14 {:.1f})">{}</text>'.format(
            (self.mt + y_end) / 2, (self.mt + y_end) / 2, ylabel))

    def line(self, xs, ys, color, width=2.0, dash=None, label=None, opacity=1.0):
        pts = " ".join("{:.1f},{:.1f}".format(self.X(x), self.Y(y)) for x, y in zip(xs, ys))
        d = ' stroke-dasharray="{}"'.format(dash) if dash else ""
        self.parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-opacity="{}"{} stroke-linejoin="round"/>'.format(
            pts, color, width, opacity, d))
        if label:
            self.legend_items.append((label, color, dash))

    def dot(self, x, y, color, r=4, label=None):
        self.parts.append('<circle cx="{:.1f}" cy="{:.1f}" r="{}" fill="{}"/>'.format(self.X(x), self.Y(y), r, color))
        if label:
            self.parts.append('<text x="{:.1f}" y="{:.1f}" font-size="11" fill="{}">{}</text>'.format(self.X(x) + 7, self.Y(y) - 7, color, label))

    def arrow(self, x, y, yaw, length, color):
        x2, y2 = x + length * math.cos(yaw), y + length * math.sin(yaw)
        self.parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" stroke-width="2.5" marker-end="url(#ah)"/>'.format(
            self.X(x), self.Y(y), self.X(x2), self.Y(y2), color))

    def text(self, x, y, s, color="#1d2330", size=11):
        self.parts.append('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}">{}</text>'.format(self.X(x), self.Y(y), size, color, s))

    def save(self, path, legend_xy=None):
        lx, ly = legend_xy or (self.ml + 12, self.mt + 16)
        for i, (label, color, dash) in enumerate(self.legend_items):
            y = ly + i * 18
            d = ' stroke-dasharray="{}"'.format(dash) if dash else ""
            self.parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2.5"{}/>'.format(lx, y, lx + 26, y, color, d))
            self.parts.append('<text x="{}" y="{}" font-size="12" fill="#1d2330">{}</text>'.format(lx + 32, y + 4, label))
        self.parts.insert(1, '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
                             '<path d="M0,0 L8,4 L0,8 z" fill="#111827"/></marker></defs>')
        self.parts.append("</svg>")
        Path(path).write_text("\n".join(self.parts), encoding="utf-8")


def position_blend(old, new, robot, blend_m=0.5, ds=0.05):
    """The first splice tried: interpolate positions instead of headings."""
    robot = np.asarray(robot, dtype=float)
    new_r = new.remainder(new.project(robot), ds)
    grid = np.arange(0.0, new_r.length + 0.5 * ds, ds)
    alpha = _smoothstep(grid / blend_m)[:, None]
    pts = new_r.point(grid)
    old_r = old.remainder(old.project(robot), ds)
    a_old = _smoothstep(grid / min(blend_m, old_r.length))[:, None]
    pts = (1 - a_old) * old_r.point(np.minimum(grid, old_r.length)) + a_old * pts
    pts = pts + (1 - alpha) * (robot - pts[0])
    return _from_points(pts, new.end_yaw, 0.0)


def max_turn_deg(path):
    d = np.diff(path.xy, axis=0)
    d = d[np.hypot(*d.T) > 1e-6]
    a = np.arctan2(d[:, 1], d[:, 0])
    return float(np.degrees(np.max(np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))))


def fig_splice():
    old = resample([(0, 0), (2.2, 0)], 0.0, 0.05)
    new = resample([(0, 0), (2 * math.cos(0.5), 2 * math.sin(0.5))], 0.5, 0.05)
    robot = (0.4, 0.0)
    pos = position_blend(old, new, robot)
    head = splice(old, new, robot, blend_m=0.5)
    p = Plot(760, 400, (-0.1, 2.3), (-0.25, 1.15), "两次预测的拼接：按位置插值 vs 按航向插值（夹角 29°，推理期间已走 0.4 m）",
             "x (m)", "y (m)", equal=True)
    p.line(old.xy[:, 0], old.xy[:, 1], COLORS["old"], 2, "6,4", "旧路径（正在跟踪）")
    p.line(new.xy[:, 0], new.xy[:, 1], COLORS["new"], 2, "6,4", "新预测（锚定在拍照时刻位姿）")
    p.line(pos.xy[:, 0], pos.xy[:, 1], COLORS["pos"], 2.5, None,
           "按位置插值：每 5 cm 最大折转 {:.1f}°".format(max_turn_deg(pos)))
    p.line(head.xy[:, 0], head.xy[:, 1], COLORS["head"], 3, None,
           "按航向插值（采用）：每 5 cm 最大折转 {:.1f}°".format(max_turn_deg(head)))
    p.dot(0, 0, COLORS["new"], 4, "拍照时刻位姿")
    p.dot(*robot, COLORS["robot"], 5, "回复到达时的机器人")
    p.dot(*new.xy[-1], COLORS["new"], 4, "VLA 预测终点")
    p.save(ASSETS / "fig_splice.svg", legend_xy=(p.ml + 14, p.mt + 18))


def stop_and_go_profile(seed=0, dt=0.05):
    rng = random.Random(seed)
    pose, t, streak = (0.0, 0.0, 0.0), 0.0, 0
    ts, vs, xs, ys = [0.0], [0.0], [0.0], [0.0]
    while t < 200:
        wps = nsim.fake_vla(pose, rng)
        lat = nsim.latency(rng)
        ts += [t, t + lat]
        vs += [0.0, 0.0]
        t += lat
        streak = streak + 1 if suggests_arrival(wps) else 0
        if streak >= 2:
            break
        v, w, duration = nsim.arc_to(*wps[1][:2])
        n = max(1, int(round(duration / dt)))
        for _ in range(n):
            pose = integrate_unicycle(pose, v, w, duration / n)
            t += duration / n
            ts.append(t)
            vs.append(v)
            xs.append(pose[0])
            ys.append(pose[1])
        ts.append(t)
        vs.append(0.0)
    return np.array(ts), np.array(vs), np.array(xs), np.array(ys)


def fig_runs():
    nav, log = nsim.run_overlapped(seed=1)
    t = np.array([r[0] for r in log])
    v = np.array([r[1] for r in log])
    xy = np.array([r[3][:2] for r in log])
    ts, vs, sx, sy = stop_and_go_profile(seed=1)

    p = Plot(560, 460, (-0.3, 3.6), (-0.4, 2.9), "L 形路线上的实际轨迹", "x (m)", "y (m)", equal=True)
    p.line([0, 3, 3], [0, 0, 2.5], COLORS["route"], 8, None, "参考路线", opacity=0.35)
    p.line(sx, sy, COLORS["sg"], 2, "5,3", "stop-and-go（{:.0f} s）".format(ts[-1]))
    p.line(xy[:, 0], xy[:, 1], COLORS["head"], 2.5, None, "并行推理（{:.0f} s）".format(t[-1]))
    p.dot(0, 0, COLORS["robot"], 4, "起点")
    p.dot(3, 2.5, COLORS["robot"], 4, "终点")
    p.save(ASSETS / "fig_trajectory.svg", legend_xy=(p.ml + 12, p.mt + 18))

    q = Plot(760, 300, (0, max(ts[-1], t[-1]) + 1), (-0.02, 0.36), "指令线速度随时间变化（同一组延迟与预测噪声）",
             "时间 (s)", "v (m/s)")
    q.line(ts, vs, COLORS["sg"], 1.6, None, "stop-and-go：等推理 → 开到航点 → 停")
    q.line(t, v, COLORS["head"], 2.5, None, "并行推理：边走边推理")
    q.save(ASSETS / "fig_speed.svg", legend_xy=(q.X(22.3), q.mt + 18))
    return t[-1], ts[-1], float(v[(v > 0.05)].mean())


def fig_labels():
    """Dense teleoperation poses -> deployment keyframes -> target of one (off-keyframe) frame."""
    import navgpt_sft.episodes as ep

    poses, pose = [], (0.0, 0.0, 0.0)
    for i in range(100):                       # 0.04 m per frame; turn left in the middle
        w = 0.9 if 40 <= i < 57 else 0.0
        pose = integrate_unicycle(pose, 0.4, w, 0.1)
        poses.append(pose)
    poses = np.asarray(poses)
    keys = ep.keyframe_indices(poses, 0.25, math.radians(15))
    current = keys[5] + 3                     # a frame between two keyframes
    wp = ep.future_waypoints(poses, current, 0.25, math.radians(15))
    origin = poses[current]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    world = np.array([[origin[0] + c * x - s * y, origin[1] + s * x + c * y] for x, y, _ in wp])

    p = Plot(760, 430, (-0.2, 3.4), (-0.45, 2.6), "训练标签：稠密轨迹 → 部署规则关键帧 → 当前帧的 8 个目标航点",
             "x (m)", "y (m)", equal=True)
    p.line(poses[:, 0], poses[:, 1], COLORS["route"], 2, None, "遥操作稠密轨迹（每帧一个位姿）")
    for k in keys[: keys.index(next(k for k in keys if k > current))]:
        p.dot(poses[k, 0], poses[k, 1], COLORS["old"], 3.5)
    p.line([], [], COLORS["old"], 0.1, None, "关键帧：每 0.25 m 或 15° 一个，构成视觉历史")
    p.line(world[:, 0], world[:, 1], COLORS["head"], 2.5, None, "8 个目标航点（相对当前帧，按同一规则往后取）")
    for i, (x, y) in enumerate(world):
        p.dot(x, y, COLORS["head"], 4, str(i + 1) if i in (0, 3, 7) else None)
    p.dot(origin[0], origin[1], COLORS["pos"], 5)
    p.text(origin[0] - 0.55, origin[1] - 0.25, "当前帧（两个关键帧之间）", COLORS["pos"], 12)
    p.save(ASSETS / "fig_labels.svg", legend_xy=(p.ml + 14, p.mt + 18))


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_labels()
    fig_splice()
    t_async, t_sg, v_mean = fig_runs()
    print("written to {}: overlapped {:.1f} s (mean {:.2f} m/s while moving), stop-and-go {:.1f} s".format(
        ASSETS, t_async, v_mean, t_sg))
