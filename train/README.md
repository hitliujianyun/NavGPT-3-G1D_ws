# navgpt_sft：NavGPT VLA 的 LoRA 微调

在官方 NavGPT3-4B/8B 权重上做后训练：**语言模型加 LoRA，动作头全量训练，视觉编码器冻结**（嵌入层和 LM head 也冻结）。训练好的结果合并成完整的 NavGPT3 checkpoint，`deploy/g1d/server/serve_vla.py --checkpoint` 可以直接加载，部署侧不用改代码。

损失按论文 Eq. 5：导航样本对 24 维归一化航点做 MSE，视觉语言样本只算 assistant 回答部分的 next-token 交叉熵，两者权重 1:1。

## 安装

GPU 服务器上，`scripts/setup_gpu_server.sh` 已经把本包和 PEFT 装进 `code/NavGPT-3/.envs/vla`。已有环境的话补装：

```bash
uv pip install --python code/NavGPT-3/.envs/vla/bin/python --editable train
```

下面的命令都在工作区根目录、用这个环境的 python 运行（简写为 `python`）。

## 数据格式

导航数据每个 episode 占一行 JSON，图像路径相对 manifest 所在目录：

```json
{"episode_id": "g1d_0001",
 "instructions": ["Leave the lab, turn left and stop at the meeting room door.", "..."],
 "frames": [{"image": "frames/00000.jpg", "pose": [x, y, yaw]}, ...]}
```

- `pose` 是拍照时刻的里程计位姿：x、y 单位米，yaw 单位弧度，逆时针为正。
- 帧可以是稠密的（遥操作按几 Hz 录制），也可以是已经按关键帧取好的。训练时统一按部署规则（移动 0.25 m 或转动 15°）取关键帧作为视觉历史。
- 标签是从当前帧往后、按同一规则走出的 8 个累计位姿，表达在当前机体系（x 向前、y 向左、θ 逆时针），不足 8 个用终点补齐，表示停止。再除以 `nav.scale`（默认 habitat_nav，与 `serve_vla.py` 一致）。
- 图像必须与部署时 VLA 看到的完全一致：teleimager 左目、居中裁方、400×400，即 `navgpt_g1d.camera.preprocess` 的输出。
- 多相机数据（`nav.view_num: 4`）每帧写 `"images": [front, right, back, left]`。

视觉语言数据（`kind: vl`，只训练 next-token）：

```json
{"images": ["img/0001.jpg"],
 "conversations": [{"role": "user", "content": "<image>\nWhere is the trash can?"},
                   {"role": "assistant", "content": "Left of the door."}]}
```

把部署日志里**成功的**自主运行转成训练数据（默认只保留 `arrived` 的运行，转换后请人工复核）：

```bash
python -m navgpt_sft from-g1d-logs deploy/g1d/logs --out data/sft/g1d_runs/train.jsonl
```

训练前检查数据：每个 manifest 的 episode 数、样本数、停止/转弯/直行分布、缺失图像数，以及随机样本的历史帧和航点（米、米、度）：

```bash
python -m navgpt_sft inspect --config train/configs/g1d_lora_4b.yaml
```

## 训练

```bash
# 单卡（4B 在 RTX 5090 32 GB 上，batch 1）
python -m navgpt_sft train --config train/configs/g1d_lora_4b.yaml

# 多卡（DDP，每卡一份完整模型），8B 建议用 80 GB 卡
accelerate launch --num_processes 2 -m navgpt_sft train --config train/configs/g1d_lora_8b.yaml

# 临时改参数；从断点继续
python -m navgpt_sft train --config train/configs/g1d_lora_4b.yaml --set train.epochs=1 --set lora.r=16
python -m navgpt_sft train --config train/configs/g1d_lora_4b.yaml --set train.resume=outputs/sft/g1d_lora_4b/checkpoint-500
```

`train.output_dir` 下的内容：

- `log.jsonl`：训练损失、学习率、显存，以及每 `eval_steps` 一次的验证指标。
- `checkpoint-<step>/`：`adapter/`（PEFT LoRA）、`action_head.safetensors`、优化器状态、配置。不含基座权重。

## 评估与合并

```bash
# 离线指标：首/末航点位置误差 (m)、航向误差 (deg)、到达判断的 precision/recall
python -m navgpt_sft eval --config train/configs/g1d_lora_4b.yaml                         # 基座模型，作基线
python -m navgpt_sft eval --config train/configs/g1d_lora_4b.yaml --adapter outputs/sft/g1d_lora_4b/checkpoint-1200

# 合并成完整 NavGPT3 checkpoint（通过上游 validate_checkpoint），然后照常部署
python -m navgpt_sft merge --config train/configs/g1d_lora_4b.yaml \
    --checkpoint outputs/sft/g1d_lora_4b/checkpoint-1200 --out models/NavGPT3-4B-g1d
CHECKPOINT=models/NavGPT3-4B-g1d GPU=0 deploy/g1d/scripts/start_vla_server.sh
```

## 与推理保持一致的地方

| 项 | 做法 |
|---|---|
| Prompt | 直接调用上游 `NavGPTVLA.PROMPT_TEMPLATE`、`_build_image_describe` 和 `build_prompt` |
| 图像与 token 分配 | 上游 `NavGPT3Processor`（PIL 后端），走与 `frame_cache=on_demand` 相同的逐图预算路径；默认用部署值（3072 / γ=2 / codec），30% 的样本随机化预算 |
| 历史采样 | 最多 16 帧，保留首帧和当前帧，中间均匀抽取（同 `sample_indices`） |
| 当前帧 | 50% 的样本取两个关键帧之间的帧（对应部署中 `commit=false` 的请求），需要稠密帧 |
| 动作输出 | 训练 forward 与 `predict_actions` 相同（最后一个 prompt token → 动作头）；测试验证两者在 bf16 舍入范围内一致 |
| 停止 | 终点补齐的全零航点；按类别重采样（停止、转弯各 2 倍） |

## 测试

```bash
cd train && python -m unittest discover -s tests
```

- 纯 Python 测试：关键帧、航点坐标系与补齐、历史采样、manifest、配置、日志转换。
- 端到端测试（`test_pipeline.py`）：在 CPU 上构建一个随机初始化的小 NavGPT3，依次跑训练、评估、断点续训、合并，再用上游严格加载器加载，并检查合并结果与 adapter 输出一致。它需要 torch、torchvision、transformers、peft，以及 Qwen3-VL 的 tokenizer/processor 文件（默认放在 `models/_qwen3vl_tok`，见 `tests/tiny_checkpoint.py`），缺少时自动跳过。

## 文件

| 文件 | 内容 |
|---|---|
| `navgpt_sft/episodes.py` | episode 格式、关键帧、航点标签、历史采样（纯 numpy） |
| `navgpt_sft/data.py` | 导航和视觉语言样本编码、混合采样器、padding |
| `navgpt_sft/model.py` | 冻结与 LoRA 注入、训练 forward 与损失、adapter 和动作头的保存/加载 |
| `navgpt_sft/trainer.py` | 训练循环（Accelerate）、离线评估、checkpoint 与续训 |
| `navgpt_sft/export.py` | 合并为完整 NavGPT3 checkpoint |
| `navgpt_sft/g1d_logs.py` | navgpt_g1d 运行日志转 manifest |
| `navgpt_sft/cli.py` | 命令行 |
| `configs/` | 4B（单卡 5090）与 8B（80 GB 卡）配置 |
