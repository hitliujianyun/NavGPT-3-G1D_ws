# 宇树 G1-D 部署相关资料（存档，抓取于 2026-11-20）

来源：
- 产品页 https://www.unitree-robot.com/cn/G1-D/
- 开发者文档 https://support.unitree.com/home/zh/G1D_Developer
- unitree_sdk2（已克隆到 code/unitree_sdk2）：include/unitree/robot/g1/agv/、example/g1/g1d/
- teleimager（已克隆到 code/teleimager）：头部相机图像服务
- IT之家 2025-11-13 报道 https://www.ithome.com/0/897/148.htm

## 1. 整机参数

| 项目 | G1-D 通用版 | G1-D 旗舰版 |
|---|---|---|
| 尺寸（立柱最低/最高） | 约 1260/1680 × 500 × 500 mm | 约 1260/1680 × 525 × 570 mm |
| 重量（含电池） | 约 50 kg | 约 80 kg |
| 整机自由度（不含末端） | 17 | 19 |
| 单臂自由度 / 负载 | 7 / 约 3 kg | 7 / 约 3 kg |
| 腰部 | 2 DoF，Z ±155°，Y -2.5°~+135° | 同左 |
| 立柱升降 | 约 60 mm/s（SDK 注释最大 76.5 mm/s） | 同左 |
| 底盘 | 无（推拉式，无电驱） | 双轮差速，最大 1.5 m/s，可原地 360° 旋转 |
| 底盘传感器 | / | 激光雷达×1 + 深度相机×2 + 物理碰撞传感器×2 + 低矮障碍物传感器×2 |
| 感知 | 头部高清双目相机×1 + 腕部高清相机×2 | 同左 |
| 头部雷达 | Livox MID-360（水平 360°，竖直最大 59°） | 同左 |
| 开发算力（PC2） | Jetson Orin NX 16GB（100 TOPS），192.168.123.164，用户 unitree / 密码 123 | 同左 |
| 电池 / 续航 | 上身 9 Ah / 约 2 h | 底盘 30 Ah / 约 6 h |

**结论：只有 G1-D 旗舰版带电动底盘，能做导航部署。** 运控计算单元不对外开放，二次开发只能用 PC2。

### 底盘硬件规格速查（开发者文档）

| 项目 | 规格 |
|---|---|
| 底盘尺寸 | 557 × 525 mm，旋转直径 666 mm |
| 净重 / 额定负载 | 63 kg / 40 kg |
| 最大速度 / 爬坡 | 1.5 m/s / 10°（≤18%） |
| 驱动 | 2 × 6.5 寸轮毂电机 + 4 × 2.5 寸万向轮 |
| 激光雷达（ToF S2） | 最大扫描半径 30 m |
| 深度相机 | 2 个，0.3 ~ 3.5 m，FOV H75° / V51° |
| 超声波 | 2 个，最大 50 cm |
| 接口 | 千兆网口、DC 48V/30A 输出、2.4 GHz Wi-Fi |

## 2. 底盘控制接口（unitree_sdk2，C++）

```cpp
// include/unitree/robot/g1/agv/g1_agv_api.hpp
const std::string AGV_SERVICE_NAME = "agv";
const std::string AGV_API_VERSION  = "1.0.0.1";
const int32_t ROBOT_API_ID_AGV_MOVE          = 1001;
const int32_t ROBOT_API_ID_AGV_HEIGHT_ADJUST = 1002;
// MoveParameter JSON: {"vx":..., "vy":..., "vyaw":...}

// include/unitree/robot/g1/agv/g1_agv_client.hpp
int32_t Move(float vx, float vy, float vyaw);
//   vx   ∈ [-1.5, 1.5] m/s，正值前进
//   vy   不支持（AGV 忽略），传 0
//   vyaw ∈ [-0.6, 0.6] rad/s，正值逆时针
//   非阻塞，立即返回
int32_t HeightAdjust(float vz);  // vz ∈ [-1, 1]，×76.5 mm/s
```

示例 `example/g1/g1d/g1_agv_client_example.cpp`：`ChannelFactory::Instance()->Init(0)` → `AgvClient ac; ac.SetTimeout(3.0f); ac.Init(); ac.Move(vx, 0, vyaw)`，每 50 ms 发送一次。

立柱高度反馈：DDS 话题 `rt/hispeed_state`（`geometry_msgs::msg::dds_::Point32_`，`y()` 为当前高度），示例 `g1d_height_control.cpp` 用 PD（kP=20, kD=0.3）。

**unitree_sdk2_python 没有 AgvClient**，本项目在 `deploy/g1d/navgpt_g1d/agv.py` 用通用 RPC `Client` 按上述常量实现了 Python 版。

## 3. 里程计（待真机确认）

第三方 G1（足式）DDS 话题整理（neon 项目）：`rt/odommodestate`（SportModeState_，含 position/velocity/imu）有数据，`rt/odom` 声明但无数据；另有 `rt/unitree_slam/odom`、`rt/utlidar/cloud_livox_mid360`（10 Hz 点云）。
**这些是 G1 足式版的观察结果，G1-D 底盘的里程计话题需要在真机上用 DDS 发现确认。**

## 4. 头部相机取流（teleimager）

- PC2 上运行 `python -m teleimager.image_server`（或 `teleimager-server`），配置 head_camera：`type: uvc`，`binocular: true`，`image_shape: [480, 1280]`（左右目并排），`fps: 30`，`enable_zmq: true`，`zmq_port: 55555`，WebRTC 端口 60001。
- 协议：ZMQ PUB，每条消息就是一帧 JPEG 字节（CONFLATE，只保留最新帧）；配置查询走 REQ/REP 端口 60000（发送 `GET_DATA`）。
- 浏览器测试：`https://192.168.123.164:60001` 点 Start。
- **头部相机的视场角、分辨率官方未公布，需实测。**
