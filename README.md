# NavGPT-3 工作区（目标：部署到宇树 G1-D）

主文档：[docs/NavGPT-3_详细解读与G1-D部署方案.html](docs/NavGPT-3_详细解读与G1-D部署方案.html)（论文解读、代码解读、仿真复现、G1-D 部署方案）

部署工程：[deploy/g1d/README.md](deploy/g1d/README.md)（通过 `rt/cmd_vel_no_limit` 控制底盘、并行推理、平滑轨迹的完整 VLN 执行工程）

| 目录 | 内容 |
|---|---|
| `docs/papers/` | 论文 PDF（arXiv:2610.10787 v2） |
| `docs/web/` | 网页存档：论文全文文本、官方 README、项目页与模型卡、G1-D 资料 |
| `docs/assets/` | 文档引用的论文插图 |
| `code/` | 上游仓库，保持原样：`NavGPT-3`、`unitree_sdk2`、`unitree_sdk2_python`、`teleimager` |
| `deploy/g1d/` | G1-D 部署工程 navgpt_g1d（机器人侧运行时 + 5090 上的 VLA 服务 + 仿真 + 测试） |
| `train/` | VLA 后训练 navgpt_sft（LoRA 语言模型 + 动作头全训 + 视觉编码器冻结），合并后可直接部署，见 [train/README.md](train/README.md) |
| `scripts/` | GPU 服务器环境安装（自动适配 RTX 5090）、权重下载 |
| `models/` | 模型权重（GPU 服务器上下载，不入库） |
| `data/` | MP3D / R2R-CE / RxR-CE 数据（不入库） |

更新上游代码：`git -C code/NavGPT-3 pull`（官方尚未发布基于 ZEOS 的 Unitree 部署代码，发布后在此跟进）。

运行部署工程的测试与仿真：

```bash
cd deploy/g1d
python -m unittest discover -s tests
python -m navgpt_g1d sim --route L --crossing
```
