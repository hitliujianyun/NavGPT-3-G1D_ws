# navgpt_g1d：NavGPT-3 VLA 在宇树 G1-D 上的完整部署工程

通过 `rt/cmd_vel_no_limit`（或 g1d_ws 安全守护的 `rt/nav/cmd_vel`）控制 G1-D 差速底盘，推理与运动并行，完成 VLN 任务。

```
RTX 5090 主机 (Ubuntu 24.04)                   G1-D 机器人局域网 (PC2)
┌──────────────────────────┐   Wi-Fi   ┌────────────────────────────────────────────┐
│ server/serve_vla.py :8100│◀─ 帧+seq ─│ 推理线程: 最新帧 + 拍照时刻位姿 → /act        │
│ NavGPTVLA(view_num=1)    │── 8 航点 ─▶│   → 锚定到里程计坐标 → 与当前路径平滑拼接     │
│ NavGPT3-8B               │           │ 控制线程 20 Hz: 激光安全 → 纯追踪 → 运动授权   │
└──────────────────────────┘           │   → 监督(遥控器/急停/守护状态)                 │
                                       │   → CmdVelStreamer 30 Hz → 底盘话题           │
                                       │ DDS: slamware odom / scan   ZMQ: teleimager  │
                                       │ HTTP 控制接口 :8124 (/episode /stop /status)  │
                                       └────────────────────────────────────────────┘
```

## 两条上机路线

| | 第一天：网线 + 全在 5090 主机 | 长期：PC2 常驻 |
|---|---|---|
| 网络 | 主机网线接入机器人交换机，静态 IP 192.168.123.222 | 5090 走 Wi-Fi，只传“帧上行 / 8 航点下行” |
| 安装 | `scripts/host_quickstart.sh`（PC2 上什么都不装） | `scripts/pack_for_robot.py` 打包 → `pc2/install.sh` |
| 配置 | `config/host_wired.yaml` | `/opt/navgpt_g1d_site.yaml`（首次安装由 `g1d.yaml` 生成） |
| 适合 | 标定符号、验证接口、调参 | 无线任务、Web/Planner 调用、开机自启 |

### 第一天：网线直连

```bash
# 5090 主机，网卡 enp5s0 接机器人交换机（--set-ip 用 nmcli 配 192.168.123.222/24）
deploy/g1d/scripts/host_quickstart.sh enp5s0 --set-ip
CHECKPOINT=models/NavGPT3-8B GPU=0 deploy/g1d/scripts/start_vla_server.sh      # 另一个终端

cd deploy/g1d && PY=.venv-host/bin/python
$PY -m navgpt_g1d check --config config/host_wired.yaml     # 话题频率、相机端口、cmd_vel 写者数、VLA
$PY -m navgpt_g1d smoke --config config/host_wired.yaml     # 低速前进/左转，给出符号建议
$PY -m navgpt_g1d run   --config config/host_wired.yaml --dry-run --instruction "Walk forward and stop at the door."
$PY -m navgpt_g1d run   --config config/host_wired.yaml --instruction "Walk forward and stop at the door."
$PY -m navgpt_g1d report logs/host_wired/<时间戳>
```

### 长期：装到 PC2

```bash
# 开发机（Windows/Linux 均可；只打包已提交内容，自带 unitree_sdk2_python）
python deploy/g1d/scripts/pack_for_robot.py
scp dist/navgpt_g1d-<commit>.tar.gz unitree@192.168.123.164:~

# PC2 (JetPack 5.1.1 / Python 3.8)
tar xzf navgpt_g1d-<commit>.tar.gz
sudo navgpt_g1d/pc2/install.sh            # 安装到 /opt/navgpt_g1d，跑自测
sudoedit /opt/navgpt_g1d_site.yaml        # 填 vla.url（5090 的 Wi-Fi IP）、符号等
sudo navgpt_g1d/pc2/install.sh --enable   # 装 systemd 服务并开机自启
journalctl -u navgpt-g1d -f
```

`install.sh` 依次尝试 `--python` 指定的解释器、`/usr/bin/python3`、conda 环境 `tv`（已有 teleimager 的那个），选第一个依赖齐全的；都不齐则在 `/usr/bin/python3` 上 pip 补装。重复安装会把旧版本备份为 `/opt/navgpt_g1d.old-<时间>`，站点配置不覆盖。服务以 `unitree` 用户运行，绑定 CPU 1（g1d_ws 占用 2–7），`KillSignal=SIGINT` 让 `systemctl stop` 走正常的零速保持再释放。DDS 由 `pc2/cyclonedds.xml` 固定在 eth0，避免走 Wi-Fi。

## 底盘输出：direct 与 guard

| `base.output` | 写哪个话题 | 安全由谁负责 | 前提 |
|---|---|---|---|
| `direct`（默认） | `rt/cmd_vel_no_limit` | 本工程：激光减速/停止区、遥控器接管、碰撞/悬崖急停 | 启动时检查该话题**没有其他写者**（`base.other_writers: refuse`） |
| `guard` | `rt/nav/cmd_vel` | g1d_ws 安全守护（Mid-360 三维障碍、急停、手动接管）+ 本工程激光 | 守护已运行；启动时等 3 s 内收到 `rt/nav/guard_state` |

g1d_ws 的守护是 `rt/cmd_vel_no_limit` 的唯一写者，**有它在时必须用 `guard`**，否则 direct 模式会因检测到第二个写者而拒绝启动。底盘激光只有约 ±133° 有效、看不到悬空/低矮物体和玻璃，正式无人值守运行建议 guard 模式。guard 模式下还可以把 `topics.odom` 换成 `rt/nav/base_odom`（LIO + 轮速融合，20 Hz）。

守护状态的处理：

- `estop` / `manual` → 任务立即结束，状态为 `estop` / `manual_override`（手动接管时不做零速保持，通道立刻交给遥控器）。
- `blocked` → 守护在限速，本工程继续跟踪、暂停“卡住”判定；超过 `supervisor.guard_block_timeout_s`（20 s）结束为 `blocked`。
- 超过 0.5 s 没有新的守护状态 → `guard_lost`，停车结束。

## 监督与急停（direct 模式）

| 来源 | 触发 | 结果 |
|---|---|---|
| 遥控器 `rt/wirelesscontroller` | 任一摇杆超出死区 0.15 | `manual_override`，立即释放通道 |
| 底盘 HTTP `/api/platform/v1/events` | 启动后出现 BUMPER_TRIGGERED / CLIFF_DETECTED | `estop`，锁存，需人工清除 |

清除锁存：`curl -X POST http://<机器人IP>:8124/estop/clear`（任务运行中返回 409）。`check` 会显示遥控器与底盘事件接口是否在线。

## 相机端口自动发现

`camera.port: null` 时向 teleimager 的配置端口（60000，`GET_DATA`）请求相机配置，读取 `head_camera` 的 ZMQ 端口与是否双目（新旧两种配置格式都支持）；在配置里写死的值优先。`check` 会打印最终使用的端口和图像尺寸。

## 示教录制：产出 navgpt_sft 训练数据

```bash
python -m navgpt_g1d record --config config/g1d.yaml --instruction "Go to the trash can."
# Enter 开始 → 用遥控器开过去 → Enter 结束 → 确认保留
```

以 4 Hz 存头部相机帧，每帧配“拍照时刻”的里程计位姿（按 `camera.latency_s` 回推），写到 `data/sft/g1d_teleop/<episode>/00000.jpg …`，并向 `train.jsonl` 追加一行，可直接被 `train/navgpt_sft` 的 `load_episodes` 读取。少于 2 帧或没有指令的 episode 不写入。

## 关键机制

| 问题 | 做法 | 代码 |
|---|---|---|
| 推理期间机器人已移动 | 航点用**拍照时刻**位姿锚定到里程计坐标 | `pose_buffer.py`、`path.path_from_waypoints` |
| 两次推理结果接续 | 序号去重、过期丢弃；裁掉已走部分；0.5 m 内**按航向**过渡；终点误差沿全程分摊 | `navigator.py`、`path.splice` |
| 历史帧间距与训练一致 | 每 0.25 m / 15° 才提交一帧进 VLA 历史，其余帧只当“当前观测” | `navigator.next_request`、`serve_vla.act_without_commit` |
| 轨迹平滑 | 纯追踪 + 曲率/刹停/最大速度三重限速 + 加速度限幅；中间航点不停不对齐 | `tracker.py` |
| 无线卡顿 | 1.5 s 无回复限速 0.1 m/s，3 s 停车；路径缓冲约 1 m | `navigator.control` |
| 安全 | 激光减速区按距离限速；停止区随速度伸缩，触发保护停止（立即置零）；区域清空 1 s 后重新授权，只用恢复后拍的帧 | `safety.py`、`authority.py`、`episode.py` |
| 外部监督 | 守护状态 / 遥控器 / 碰撞悬崖事件，统一成“中止 / 受阻 / 正常” | `supervisor.py`、`chassis_events.py` |
| 底盘通道 | 30 Hz 发布；指令超过 0.3 s 未更新自动发零；结束时零速保持 1 s（guard 模式 0.2 s）后释放 | `dds.CmdVelStreamer` |
| DDS 网卡 | 按 IP 绑定网卡（`dds.interface_ip`），或 `CYCLONEDDS_URI` | `dds.init_dds`、`pc2/cyclonedds.xml` |

速度上限的经验规则：`tracker.v_max ≤ navigator.keyframe_m × 实测推理频率`（报告里的 `committed_frames` 与 `replies` 可辅助判断）。

## 上线前必须实测的项

- 停发指令后底盘多久停：g1d_ws `tools/cmd_vel_probe.py`，要求 0.6 s 内停住。本工程 0.3 s 无新指令即发零，但进程被杀时是否停车取决于底盘固件。
- 速度/里程计符号：`smoke` 给出建议，写入 `base.cmd_x_sign / cmd_yaw_sign / odom_yaw_sign`。
- 话题与消息类型：g1d_ws `tools/check_interfaces.py` 或本工程 `check`。LaserScan 类型名 `sensor_msgs.msg.dds_.LaserScan_` 已在实机上确认；slamware odom 的时间戳有约 181 s 偏移，本工程一律用接收时刻。

## 测试

```bash
cd deploy/g1d && python -m unittest discover -s tests
```

包括路径拼接、跟踪器、导航调度的运动学仿真，安全层与监督逻辑，速度发布器，相机配置发现（含真实 ZMQ 往返），示教录制（产出的数据能被训练工程读取），真实 DDS 回环（写者检测、direct 拒绝第二写者、guard 模式路由与守护丢失），以及多线程端到端仿真（平滑到达、行人横穿后恢复、遥控接管、守护限速与超时）。

## 文件

| 文件 | 内容 |
|---|---|
| `navgpt_g1d/episode.py` | 一次任务：推理线程 + 控制线程、安全停止/恢复、监督中止、卡住/超时/里程计丢失、日志 |
| `navgpt_g1d/navigator.py` | 无线程调度逻辑：请求序号、关键帧提交、到达判定、断供处理 |
| `navgpt_g1d/path.py` / `tracker.py` | 参考路径拼接 / 路径跟踪 |
| `navgpt_g1d/safety.py` / `authority.py` | 激光保护 / 运动授权 |
| `navgpt_g1d/supervisor.py` / `chassis_events.py` | 守护状态、遥控器、急停锁存 / 底盘事件轮询 |
| `navgpt_g1d/dds.py` / `laserscan_idl.py` | DDS：odom、scan、cmd_vel、守护状态、遥控器、写者检测；LaserScan 消息定义 |
| `navgpt_g1d/camera.py` / `vla_client.py` | teleimager 头部相机与端口发现 / VLA 客户端 |
| `navgpt_g1d/robot.py` / `sim.py` | 真机（direct/guard、启动前检查）/ 仿真机器人（同一接口） |
| `navgpt_g1d/record.py` | 示教录制 → navgpt_sft 数据 |
| `navgpt_g1d/api.py` / `cli.py` / `report.py` | HTTP 控制接口 / 命令行 / 日志指标 |
| `server/serve_vla.py` | 5090 上的单视角 VLA 服务 |
| `config/g1d.yaml` / `host_wired.yaml` | 全部参数 / 网线直连覆盖项 |
| `pc2/` | PC2 安装脚本、systemd 单元、环境文件、CycloneDDS 配置 |
| `scripts/pack_for_robot.py` / `host_quickstart.sh` | 打包 / 5090 主机一键环境 |
