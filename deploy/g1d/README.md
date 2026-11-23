# navgpt_g1d：NavGPT-3 VLA 在宇树 G1-D 上的完整部署工程

通过 `rt/cmd_vel_no_limit` 控制 G1-D 差速底盘，推理与运动并行，完成 VLN 任务。

```
RTX 5090 主机 (Ubuntu 24.04)                   G1-D 机器人局域网 (PC2)
┌──────────────────────────┐   Wi-Fi   ┌────────────────────────────────────────────┐
│ server/serve_vla.py :8100│◀─ 帧+seq ─│ 推理线程: 最新帧 + 拍照时刻位姿 → /act        │
│ NavGPTVLA(view_num=1)    │── 8 航点 ─▶│   → 锚定到里程计坐标 → 与当前路径平滑拼接     │
│ NavGPT3-8B               │           │ 控制线程 20 Hz: 激光安全 → 纯追踪 → 运动授权   │
└──────────────────────────┘           │   → CmdVelStreamer 30 Hz → rt/cmd_vel_no_limit │
                                       │ DDS: slamware odom / scan   ZMQ: teleimager  │
                                       │ HTTP 控制接口 :8124 (/episode /stop /status)  │
                                       └────────────────────────────────────────────┘
```

## 安装

**5090 主机**（推理）：按工作区 `scripts/setup_gpu_server.sh` 装好 `code/NavGPT-3/.envs/vla`，用 `scripts/download_weights.sh` 下载权重。

**机器人侧**（PC2 或机器人局域网内任一 Linux 主机）：

```bash
pip install -e code/unitree_sdk2_python      # 需要 CycloneDDS，见其 README
pip install -e "deploy/g1d[robot]"           # Jetson 上若已有系统 OpenCV，可只装 pyzmq
```

## 上机步骤（按顺序）

```bash
# 0. 不接机器人：全链路仿真（含行人横穿 → 保护停止 → 自动恢复）
python -m navgpt_g1d sim --route L --crossing

# 1. 5090 主机：启动 VLA 服务（启动时自动预热）
CHECKPOINT=models/NavGPT3-8B GPU=0 deploy/g1d/scripts/start_vla_server.sh

# 2. 机器人侧：填好 config/g1d.yaml 中标 verify 的项，然后自检
cd deploy/g1d
python -m navgpt_g1d check --config config/g1d.yaml     # odom/scan/相机频率、VLA 连通

# 3. 底盘方向冒烟测试：低速前进、左转，对照里程计，给出 cmd/odom 符号建议
python -m navgpt_g1d smoke --config config/g1d.yaml

# 4. 空跑：除了不发 /cmd_vel_no_limit，其余全部真实
python -m navgpt_g1d run --config config/g1d.yaml --dry-run --instruction "Walk forward and stop at the door."

# 5. 实跑
python -m navgpt_g1d run --config config/g1d.yaml --instruction "Walk forward and stop at the door."

# 6. 看指标：平均速度、停顿比例、最大加速度、推理延迟、拼接偏差
python -m navgpt_g1d report logs/<时间戳>
```

常驻服务方式（供 Web 界面或 Planner 调用）：

```bash
deploy/g1d/scripts/start_robot.sh
curl -X POST http://<机器人IP>:8124/episode -d '{"instruction": "Go to the trash can."}'
curl http://<机器人IP>:8124/status
curl -X POST http://<机器人IP>:8124/stop
```

## 关键机制

| 问题 | 做法 | 代码 |
|---|---|---|
| 推理期间机器人已移动 | 航点用**拍照时刻**位姿锚定到里程计坐标 | `pose_buffer.py`、`path.path_from_waypoints` |
| 两次推理结果接续 | 序号去重、过期丢弃；裁掉已走部分；0.5 m 内**按航向**过渡；终点误差沿全程分摊 | `navigator.py`、`path.splice` |
| 历史帧间距与训练一致 | 每 0.25 m / 15° 才提交一帧进 VLA 历史，其余帧只当“当前观测” | `navigator.next_request`、`serve_vla.act_without_commit` |
| 轨迹平滑 | 纯追踪 + 曲率/刹停/最大速度三重限速 + 加速度限幅；中间航点不停不对齐 | `tracker.py` |
| 无线卡顿 | 1.5 s 无回复限速 0.1 m/s，3 s 停车；路径缓冲约 1 m | `navigator.control` |
| 安全 | 激光减速区按距离限速；停止区随速度伸缩，触发保护停止（立即置零）；区域清空 1 s 后重新授权，只用恢复后拍的帧 | `safety.py`、`authority.py`、`episode.py` |
| 底盘通道 | 30 Hz 发布；指令超过 0.3 s 未更新自动发零；结束时零速保持 1 s 后释放给遥控器 | `dds.CmdVelStreamer` |
| DDS 网卡 | 支持按 IP 绑定网卡（`dds.interface_ip`） | `dds.init_dds` |

速度上限的经验规则：`tracker.v_max ≤ navigator.keyframe_m × 实测推理频率`（报告里的 `committed_frames` 与 `replies` 可辅助判断）。

## 测试

```bash
cd deploy/g1d && python -m unittest discover -s tests
```

包括路径拼接、跟踪器、导航调度的运动学仿真，安全层，速度发布器，配置校验，以及多线程端到端仿真（平滑到达、行人横穿后恢复）。

## 文件

| 文件 | 内容 |
|---|---|
| `navgpt_g1d/episode.py` | 一次任务：推理线程 + 控制线程、安全停止/恢复、卡住/超时/里程计丢失、日志 |
| `navgpt_g1d/navigator.py` | 无线程调度逻辑：请求序号、关键帧提交、到达判定、断供处理 |
| `navgpt_g1d/path.py` / `tracker.py` | 参考路径拼接 / 路径跟踪 |
| `navgpt_g1d/safety.py` / `authority.py` | 激光保护 / 运动授权 |
| `navgpt_g1d/dds.py` / `laserscan_idl.py` | DDS：odom、scan、cmd_vel；LaserScan 消息定义 |
| `navgpt_g1d/camera.py` / `vla_client.py` | teleimager 头部相机 / VLA 客户端 |
| `navgpt_g1d/robot.py` / `sim.py` | 真机 / 仿真机器人（同一接口） |
| `navgpt_g1d/api.py` / `cli.py` / `report.py` | HTTP 控制接口 / 命令行 / 日志指标 |
| `server/serve_vla.py` | 5090 上的单视角 VLA 服务 |
| `config/g1d.yaml` | 全部参数 |
