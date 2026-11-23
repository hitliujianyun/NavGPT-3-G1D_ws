# NavGPT-3 项目主页 / Hugging Face 模型卡 / 官方博客（存档，抓取于 2026-11-20）

## 1. 项目主页 https://metacognitionai.github.io/NavGPT3/

- 标题：Harnessing Context in a Hierarchical Navigation Runtime
- 作者：Gengze Zhou (1,2†), Yicong Hong (4), Jiazhao Zhang (5), Xunyi Zhao (1), Jian Zhou (1), Zixing Lei (6), Zun Wang (7), Chongyang Zhao (8), Xionghui Chen (5), Stephen Gould (2,3), Anton van den Hengel (1,2), Qi Wu (1,2†)
- 机构：1 AIML Adelaide University；2 Metacognition；3 ANU；4 Roblox；5 PKU；6 SJTU；7 UNC Chapel Hill；8 UNSW
- 通讯：{gengze.zhou, qi.wu01}@adelaide.edu.au

要点：
- NavGPT Planner 选择下一步并决定任务何时完成；harness 提供观测、空间工具、持久路线状态和执行反馈；runtime 控制逻辑线程、权限与运动授权（motion authority）。
- Planner 可以直接调用工具行动，也可以把长程导航委托给 NavGPT VLA，然后检查返回证据并修正路线。
- 以 GPT-6 Astra 作为 Planner 时：R2R-CE SR 76.94 → 81.51（+4.57），RxR-CE SR 82.39 → 90.43（+8.04）。
- 人类参考：英文 RxR 90.4%（人类实验为离散导航，设定与 RxR-CE 不同）。
- 网页包含可交互的轨迹浏览器（需 JavaScript），展示 Claude Opus 5 / GPT-6 Astra 在 R2R-CE、RxR-CE 上的录制对话与工具调用。

## 2. Hugging Face: Metacognition-AI/NavGPT3-8B

| 项 | 值 |
|---|---|
| Base model | Qwen3-VL-8B-Instruct |
| Parameters | 8.77B, BF16（下载约 17.6 GB） |
| Action head | 两层 MLP，作用于最后一个 prompt token 的最终隐状态；输出 8 个归一化 (x, y, θ) 航点 |
| Observations | 最多 16 个历史步 × 4 视角，400×400，120° FOV |
| Visual tokens | 所有图像共享 3,072 token 预算，按时近、相机、帧变化量分配 |
| Training data | 约 19.28M 样本：VLN、具身导航 QA、跨具身导航与 grounding |
| Code | Transformers 5.18 上的 NavGPT3 类 |

## 3. Hugging Face: Metacognition-AI/NavGPT3-4B

| 项 | 值 |
|---|---|
| Base model | Qwen3-VL-4B-Instruct |
| Parameters | 4.44B, BF16（下载约 9.7 GB） |
| 其余 | 与 8B 相同 |

用法：
```bash
hf download Metacognition-AI/NavGPT3-4B --local-dir checkpoints/NavGPT3-4B
```
```python
from navgpt.vla.runtime import load_from_pretrained
model, processor = load_from_pretrained("checkpoints/NavGPT3-4B", device="cuda")
```
- 模型需要 NavGPT-3 仓库中的 NavGPT3 类，不能仅用 `AutoModel` 加载。
- `model.predict_actions(**inputs)` 返回归一化航点；`navgpt.vla` agent 负责多视角历史、逐图 token 预算和航点反归一化（米/弧度）。

**Limitations（原文）**
- Trained and evaluated in Habitat simulation (R2R-CE, RxR-CE). Waypoint scales follow the simulated embodiment; other robots need their own scales and testing.
- Evaluated on English instructions.

**License**：权重为 **GNU AGPL-3.0**（由 Apache-2.0 的 Qwen3-VL 微调而来）；NavGPT-3 源码另以 Apache-2.0 授权。

## 4. 官方博客 Metacognition

https://metacognitionai.com/research/navgpt-3-reaches-human-level-performance-in-vision-and-language-navigation/

- 标题：NavGPT-3 reaches human-level performance in vision-and-language navigation
- RxR 上成功率 90.4%，与人类 90.4% 持平；对比 Alibaba Qwen-RobotNav 76.5%、Mistral Robostral Navigate 75.1%。
- VLN 任务由 Gould、van den Hengel、Qi Wu 等人于 8 年前提出（Anderson et al., CVPR 2018）。
- 系统由多个模型协作：一个理解指令并规划路线，另一个选择具体动作，协调系统让各组件协同工作。
- 注意：博客写 “NavGPT VLA 8B … reaching 78.2% on Room-Across-Room”，与论文 Table 6 的 82.39 SR 不一致，以论文为准。

## 5. 第三方笔记

- ImChong/Robotics_Notebooks: wiki/entities/paper-navgpt-3.md —— 中文摘要：“NavGPT-3 将语言模型规划与导航 VLA 经分层 harness 和线程运行时连接，在 VLN-CE 仿真中评测；代码与权重公开，Unitree 部署待发布。”
- Startuply 报道：ZEOS 被描述为位于大模型之下的运行时，推理/执行/监控作为独立线程运行，各有上下文与权限。
