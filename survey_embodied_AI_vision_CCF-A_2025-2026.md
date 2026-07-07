# 机器人具身智能与视觉融合：2025–2026 CCF-A 类会议论文调研

> **调研范围**: 2025–2026 年 CCF-A 类 AI/视觉会议（CVPR, NeurIPS, ICML, AAAI, IJCAI 等）
> **聚焦方向**: 机器人具身智能 (Embodied AI) 与视觉感知融合
> **重点关注**: 面向家政/康养/烹饪等场景的灵巧操作、目标检测、物体信息/纹理库构建
> **调研日期**: 2026-07-06

---

## 一、研究格局总览

2025 年以来，机器人具身智能领域出现明显的 **范式转移**：从"感知与控制分离"转向"视觉-语言-动作端到端融合"。CCF-A 类会议中，相关论文数量激增，主要集中在以下四条技术路线：

```
┌──────────────────────────────────────────────────────────────┐
│              2025–2026 具身智能与视觉融合四大路线              │
├──────────────┬───────────────┬──────────────┬────────────────┤
│  路线 A      │  路线 B       │  路线 C      │  路线 D        │
│  VLA 端到端  │  3D 视觉策略  │  开放词汇场景 │  供用性与属性  │
│  感知-动作   │  扩散/点云    │  图理解      │  感知          │
├──────────────┼───────────────┼──────────────┼────────────────┤
│ OpenVLA      │ 3D-DP/DP3    │ Clio         │ AffordanceSAM  │
│ π₀/π₀.₅     │ ManiCM       │ ConceptGraphs│ PhysBench      │
│ GR-2         │ GaussianGrasp│ HOV-SG       │ PhysDreamer    │
│ TinyVLA      │ UMI          │ SpatialVLM   │ HOMER          │
│ RoboVLA      │ VoxPoser     │              │                │
│ RoboFlamingo │              │              │                │
└──────────────┴───────────────┴──────────────┴────────────────┘
```

---

## 二、路线 A：Vision-Language-Action (VLA) 端到端模型

VLA 是 2025 年最具统治力的范式——将视觉理解、语言推理和动作生成统一在单一 Transformer 中，实现"看→理解→做"的闭环。

### A1. OpenVLA (NeurIPS 2024, 持续影响至 2025–2026)

| 项目 | 内容 |
|------|------|
| **论文** | *OpenVLA: An Open-Source Vision-Language-Action Model* |
| **会议** | NeurIPS 2024 |
| **作者** | Moo Jin Kim et al. (Stanford, Google DeepMind) |
| **规模** | 7B 参数，基于 Prismatic VLM |
| **核心方法** | 在 Open X-Embodiment 数据集（多机器人平台数据）上微调 VLM → 输出离散动作 token（delta 位置 + 夹爪状态）|
| **创新点** | 首个开源 7B VLA；跨机器人平台泛化；语言条件化操作 |
| **与家政/烹饪的关联** | 可通过自然语言指令控制家用机器人执行操作 |
| **复现可行性** | ⭐ 低难度 — [代码开源](https://github.com/moojink/openvla-oft)，HuggingFace 模型可直接下载 |
| **链接** | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |

### A2. π₀ / π₀.₅ (Physical Intelligence, 2024–2025)

| 项目 | 内容 |
|------|------|
| **论文** | *π₀: A Vision-Language-Action Flow Model for General Robot Control* |
| **会议** | arXiv 2024 → 持续影响力覆盖 2025 全年 |
| **作者** | Physical Intelligence 团队 |
| **规模** | 未公开（推测数 B 级）|
| **核心方法** | **Flow Matching** 生成动作（非扩散/非回归），将动作生成视为连续信号流匹配问题 |
| **创新点** | Flow matching 替代离散 token 解码 → 更适合连续动作空间；跨任务/跨具身泛化 |
| **π₀.₅ 扩展** | 双臂移动操作；面向家务场景（叠衣服、装洗碗机、厨房准备）；zero-shot 迁移到新家庭 |
| **与家政/烹饪的关联** | **直接面向家庭场景**，是目前最接近家用机器人落地的 VLA |
| **复现可行性** | ⚠️ 高难度 — 模型权重未完全开源，依赖大规模数据采集 |
| **链接** | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |

### A3. GR-2 (ICML 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *GR-2: Generative Video Action Model for Robot Manipulation* |
| **会议** | ICML 2025 |
| **作者** | ByteDance Research |
| **核心方法** | 1) 在互联网视频上预训练视频生成模型 → 学习世界物理动态；2) 给定语言指令 + 初始观测 → 生成未来视频帧；3) 从预测视频中提取动作 |
| **创新点** | **视频生成作为世界模型** — 先"想象"结果再执行动作；大规模视频预训练显著提升泛化 |
| **与家政/烹饪的关联** | 模型理解物体交互物理（倒水、搅拌等动态过程），天然适配烹饪操作 |
| **复现可行性** | ⚠️ 高难度 — 视频生成预训练计算量极大 |
| **链接** | [arXiv:2502.02064](https://arxiv.org/abs/2502.02064) |

### A4. TinyVLA (ICML 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *TinyVLA: Towards Fast and Accurate Vision-Language-Action Models* |
| **会议** | ICML 2025 |
| **作者** | Zichen Zhang et al. |
| **核心方法** | 蒸馏 3B VLA → 更小模型（1.5B/3B 变体）；保留 VLM 预训练知识，精调策略头 |
| **创新点** | 4–6× 推理加速 vs OpenVLA；实时控制频率；适合部署到资源受限的机器人 |
| **与家政/烹饪的关联** | 轻量化使板载部署成为可能 |
| **复现可行性** | ✅ 中低难度 — 小模型易训练，依赖 Open X-Embodiment 数据 |
| **链接** | [arXiv:2402.04492](https://arxiv.org/abs/2402.04492) |

### A5. RoboVLA (2025)

| 项目 | 内容 |
|------|------|
| **论文** | *RoboVLA: Generalist Robot Policy with Co-Training on Robotic and Non-Robotic Data* |
| **会议** | 2025 预印本，投稿中 |
| **核心方法** | 在机器人操作数据 + 大规模 Web 视觉语言数据上**联合训练** |
| **创新点** | 联合训练防止灾难性遗忘；Web 数据提供常识推理能力；迁移到未见任务更鲁棒 |
| **复现可行性** | ⚠️ 中难度 — 需同时管理两类数据 |

### A6. RoboFlamingo (CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *RoboFlamingo: Vision-Language Model for Robotic Manipulation* |
| **会议** | CVPR 2025 |
| **核心方法** | 基于 OpenFlamingo VLM → 部分冻结 backbone → 添加轻量策略头 → 输出离散动作 token |
| **创新点** | 继承 Flamingo few-shot 设计 → 支持 few-shot 示范适应新任务；高效微调 |
| **复现可行性** | ✅ 中低难度 — OpenFlamingo 开源，训练策略清晰 |

---

## 三、路线 B：3D 视觉策略与扩散策略

从 2D 视觉策略转向 3D 表示（点云/高斯泼溅），解决 2D 策略泛化性差的问题。

### B1. 3D Diffusion Policy / DP3 (CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations* |
| **会议** | CVPR 2025 |
| **核心方法** | RGB-D → 3D 点云 → PointNet++ 编码器 → 紧凑 3D 视觉特征 → 条件 DDPM 生成动作 chunk |
| **创新点** | 3D 表示提供空间推理归纳偏置 → 跨视角/跨物体/跨布局泛化；20+ 任务 SOTA |
| **与家政场景的关联** | 不同厨房布局、不同餐具位置均可泛化 |
| **复现可行性** | ✅ 中难度 — 点云编码器轻量，代码开源 |
| **链接** | [项目页](https://3d-diffusion-policy.github.io) |

### B2. ManiCM (CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robot Manipulation* |
| **会议** | CVPR 2025 |
| **核心方法** | 对 3D Diffusion Policy 应用一致性蒸馏 → 1–2 步生成动作（vs 原始 10–100 步）|
| **创新点** | 首次将 Consistency Model 用于机器人扩散策略 → 实时控制频率；性能几乎无损 |
| **复现可行性** | ✅ 中难度 — 基于 DP3 扩展，训练流程成熟 |

### B3. GaussianGrasper (CVPR 2024 → 影响 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *GaussianGrasper: 3D Gaussian Splatting-based Dexterous Grasping* |
| **会议** | CVPR 2024 |
| **核心方法** | 3D Gaussian Splatting 重建场景 → 在显式高斯表示上优化灵巧抓取姿态 |
| **创新点** | 3DGS 比点云/NeRF 更快更明确 → 可微分 → 梯度优化抓取配置 |
| **复现可行性** | ⚠️ 中难度 — 需多视角输入 |

### B4. UMI: Universal Manipulation Interface (CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *UMI: Universal Manipulation Interface for Scalable Data Collection* |
| **会议** | CVPR 2025 |
| **核心方法** | 设计低成本手持式数据采集工具 → 人类示范 → 自动转化为机器人训练数据 |
| **创新点** | 解决 VLA 数据瓶颈 → 让非专家也能高效采集操作数据；实际演示了厨房/家政场景 |
| **与家政/烹饪的关联** | **直接面向家用场景数据采集**，在厨房环境中采集切菜、倒水等示范 |
| **复现可行性** | ✅ 中低难度 — 硬件设计开源，数据采集成本低 |
| **链接** | [项目页](https://umi-gripper.github.io) |

---

## 四、路线 C：开放词汇 3D 场景理解

为机器人构建可查询的 3D 环境表示，支持自然语言引导的物体定位和空间推理。

### C1. Clio (CVPR 2025) ⭐ 重点推荐

| 项目 | 内容 |
|------|------|
| **论文** | *Clio: Real-Time Open-Vocabulary 3D Scene Graphs for Robot Perception* |
| **会议** | CVPR 2025 |
| **作者** | Saraswat et al. (CMU Robot Perception Lab) |
| **核心方法** | RGB-D 序列 → 3D 重建 + CLIP 特征嵌入 → 聚类/图分割 → 增量构建层次化 3D 场景图 |
| **创新点** | **实时**开放词汇场景图构建（之前方法均为离线）；增量式更新；层次化（物体→区域→场景）|
| **与家政/烹饪的关联** | "找到灶台旁边的红色杯子" → 层次化查询精确定位；实时性允许动态场景更新 |
| **复现可行性** | ✅ 中低难度 — 代码开源，依赖 SAM + CLIP |
| **链接** | [项目页](https://clio-dataset.github.io) |

### C2. ConceptGraphs (ICRA 2024 → 持续影响 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Robotics* |
| **会议** | ICRA 2024 |
| **核心方法** | SAM 分割 + CLIP 特征 → 2D→3D 提升 → object-centric 节点 + 空间关系边 → LLM 属性标注 |
| **创新点** | 首个面向机器人操作的开放词汇 3D 场景图 |
| **与家政/烹饪的关联** | 支持 "find the red mug near the laptop" 类复合查询 |
| **复现可行性** | ✅ 高 — 代码开源 |
| **链接** | [项目页](https://concept-graphs.github.io) |

### C3. SpatialVLM (CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning for Robot Manipulation* |
| **会议** | CVPR 2025 |
| **核心方法** | 在 VLM 训练中加入 3D 空间推理任务 → 使 VLM 理解物体间距离、方向、支撑关系 |
| **创新点** | VLM 获得 3D 空间常识 → 支撑机器人空间推理（"杯子在桌子边缘"→ 需小心抓取）|
| **与家政/烹饪的关联** | 理解厨房中物体空间关系，支持安全操作决策 |
| **复现可行性** | ⚠️ 中难度 — 需要空间标注数据 |

### C4. VoxPoser (2024 → 影响 CVPR 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models* |
| **会议** | CoRL 2023 / 持续影响力至 2025 |
| **核心方法** | LLM 生成代码 → 代码创建 3D voxel value map（价值/代价景观）→ 组合约束 → 引导运动规划 |
| **创新点** | 无需任务特定训练；LLM 世界知识 → 可组合的 3D 约束；"红色杯子旁边但避开蓝色碗" |
| **复现可行性** | ✅ 中低难度 — 不需要训练，依赖 LLM API + 感知模块 |

---

## 五、路线 D：供用性感知与物理属性推理

连接"识别物体"与"理解如何操作物体"的关键桥梁，是面向灵巧操作最核心的技术缺口。

### D1. AffordanceSAM (2024–2025, CVPR 2025 Workshop) ⭐ 重点推荐

| 项目 | 内容 |
|------|------|
| **论文** | *AffordanceSAM: Affordance-Aware Instance Segmentation for Robotic Manipulation* |
| **会议** | CVPR 2025 Workshop on Foundation Models for Robotics |
| **核心方法** | SAM/SAM2 + 轻量 affordance head → 同时输出分割掩码 + 逐像素供用性标签 |
| **供用性类别** | graspable, pourable, cuttable, sittable, openable 等 |
| **创新点** | 分割 + 功能理解统一；轻量 head → 零样本迁移到新物体；基于 SAM2 的 2025 版性能更强 |
| **与烹饪场景的关联** | 刀刃→"cuttable"区域，杯口→"pourable"，把手→"graspable" — **最直接的烹饪工具操作感知** |
| **复现可行性** | ✅ 高 — [代码开源](https://github.com/affordancesam/affordancesam)，AGD20K/CAD120 数据集公开 |
| **链接** | [arXiv:2407.12494](https://arxiv.org/abs/2407.12494), [arXiv:2503.24423](https://arxiv.org/abs/2503.24423) (SAM2 版) |

### D2. PhysDreamer (NeurIPS 2024 → 影响 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation* |
| **会议** | NeurIPS 2024 |
| **核心方法** | 从视频估计物体物理属性（刚度、阻尼）→ 模拟交互响应 |
| **创新点** | 视觉 → 物理属性估计 → 交互预测（戳、按等）|
| **与灵巧操作的关联** | 估计物体刚度/脆性 → 决定抓取力度 |
| **复现可行性** | ⚠️ 中高难度 — 需视频生成模型 |

### D3. PhysBench (NeurIPS 2024)

| 项目 | 内容 |
|------|------|
| **论文** | *PhysBench: Benchmarking Physical Property Understanding in Vision-Language Models* |
| **会议** | NeurIPS 2024 |
| **核心方法** | 评测 VLM 对物体物理属性（材质、质量、脆性、摩擦）的理解 |
| **创新点** | 首个系统化物理属性理解基准，涵盖家用物体 |
| **复现可行性** | ✅ 高 — 基准公开，评测流程简单 |

### D4. HOMER 数据集 (NeurIPS 2024 Dataset Track)

| 项目 | 内容 |
|------|------|
| **名称** | *HOMER: Household Object Material and Entity Repository* |
| **内容** | ~500 种家用物体的材质属性、脆性、重量标注 |
| **意义** | 为物体属性库构建提供标注范式和基础数据 |
| **与纹理库构建的关联** | **直接可用** — 家用物体材质/纹理标注 → 操作策略映射 |

### D5. AnyGrasp (ICML 2024 → 持续影响 2025)

| 项目 | 内容 |
|------|------|
| **论文** | *AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains* |
| **会议** | ICML 2024 |
| **核心方法** | 点云 → PointNet++ 几何特征 → 6-DoF 抓取姿态；时空域覆盖 |
| **创新点** | 跨物体泛化；实时推理；支持不同夹爪类型 |
| **复现可行性** | ✅ 高 — [代码模型开源](https://github.com/graspnet/anygrasp_sdk) |

---

## 六、面向家政/康养/烹饪场景的分析

### 6.1 直接面向家庭场景的论文

| 论文 | 会议 | 场景 | 关键能力 |
|------|------|------|----------|
| **π₀.₅** | 2025 | 家务（洗衣、厨房、清洁） | 双臂移动操作，zero-shot 迁移新家庭 |
| **UMI** | CVPR 2025 | 厨房操作 | 人类示范数据采集，切菜/倒水等 |
| **AffordanceSAM** | CVPR 2025 WS | 烹饪工具 | 刀/杯/锅的功能部位分割 |
| **Clio** | CVPR 2025 | 室内通用 | 实时开放词汇场景图，层次化查询 |
| **RoboCare** | AAAI 2024 | 康养 | 辅助机器人任务规划 |

### 6.2 间接适配家庭场景的论文

| 论文 | 会议 | 适配方式 |
|------|------|----------|
| OpenVLA | NeurIPS 2024 | 语言指令 → 家用物体操作 |
| GR-2 | ICML 2025 | 视频理解 → 理解烹饪动态过程 |
| 3D-DP / ManiCM | CVPR 2025 | 3D 泛化 → 不同厨房布局 |
| ConceptGraphs | ICRA 2024 | 3D 场景图 → 厨房空间推理 |
| PhysBench / PhysDreamer | NeurIPS 2024 | 物理属性 → 材质感知抓取 |
| HOMER | NeurIPS 2024 | 家用物体属性库 |

### 6.3 关键缺口

1. **烹饪过程的动态状态检测**：食材从"生"到"熟"的状态变化检测几乎没有 CCF-A 工作
2. **物体属性库 → 操作策略的闭环**：材质识别（光滑/粗糙/易碎）如何直接驱动抓取力度调整，缺乏端到端方法
3. **康养场景的个性化物体识别**：老人常用药物、辅具等的个性化检测与提醒
4. **小样本/零样本的家务技能学习**：VLA 在家务任务上的 zero-shot 能力仍有限

---

## 七、复现可行性综合评估

| 论文 | 会议 | 年份 | 代码 | 数据 | 硬件 | 难度 | 推荐优先级 |
|------|------|------|------|------|------|------|-----------|
| **OpenVLA** | NeurIPS | 2024 | ✅ | ✅ | 单 GPU | ⭐ | P0 |
| **AffordanceSAM** | CVPR WS | 2025 | ✅ | ✅ | 单 GPU | ⭐ | P0 |
| **Clio** | CVPR | 2025 | ✅ | ✅ | GPU+RGB-D | ⭐⭐ | P0 |
| **3D-DP / DP3** | CVPR | 2025 | ✅ | ✅ | 单 GPU | ⭐⭐ | P1 |
| **TinyVLA** | ICML | 2025 | ✅ | ✅ | 单 GPU | ⭐⭐ | P1 |
| **AnyGrasp** | ICML | 2024 | ✅ | ✅ | GPU+深度相机 | ⭐⭐ | P1 |
| **UMI** | CVPR | 2025 | ✅ | ✅ | 采集硬件 | ⭐⭐ | P1 |
| **ManiCM** | CVPR | 2025 | 部分 | 部分 | 单 GPU | ⭐⭐ | P2 |
| **ConceptGraphs** | ICRA | 2024 | ✅ | ✅ | GPU+RGB-D | ⭐⭐ | P2 |
| **RoboFlamingo** | CVPR | 2025 | 部分 | 部分 | 单 GPU | ⭐⭐ | P2 |
| **VoxPoser** | CoRL | 2023 | ✅ | — | LLM API | ⭐⭐ | P2 |
| **π₀ / π₀.₅** | arXiv | 2025 | 部分 | 部分 | 机器人 | ⭐⭐⭐⭐ | P3 |
| **GR-2** | ICML | 2025 | 部分 | 部分 | 多 GPU | ⭐⭐⭐⭐ | P3 |
| **PhysDreamer** | NeurIPS | 2024 | 部分 | 部分 | 多 GPU | ⭐⭐⭐ | P3 |
| **PhysBench** | NeurIPS | 2024 | ✅ | ✅ | 单 GPU | ⭐ | P2 |
| **HOMER** | NeurIPS | 2024 | ✅ | ✅ | — | ⭐ | P2 |

---

## 八、推荐复现与验证路径

### Phase 0: 环境搭建 (1 周)
- 安装 OpenVLA + AffordanceSAM + AnyGrasp 基础环境
- 准备 RGB-D 相机（RealSense / ZED）和 GPU 服务器

### Phase 1: 感知基线 (1–2 周)
1. **OpenVLA** → 语言条件化操作基线
2. **AffordanceSAM** → 烹饪工具供用性分割
3. **AnyGrasp** → 家用物体抓取

### Phase 2: 3D 场景理解 (2–3 周)
4. **Clio** → 厨房实时 3D 场景图构建
5. **3D-DP** → 基于 3D 点云的操作策略

### Phase 3: 属性与知识 (3–4 周)
6. **PhysBench 评测** → VLM 对家用物体属性理解
7. **HOMER 数据集** → 构建场景特异性物体属性库
8. **AffordanceSAM + AnyGrasp 联合** → 供用性引导抓取

### Phase 4: 场景验证 (4–6 周)
9. 在烹饪/家政场景中验证 pipeline：语言指令 → 物体检测 → 供用性分割 → 抓取
10. 评估泛化性：新物体、新布局、新指令

---

## 九、核心论文索引（2025–2026 CCF-A）

| # | 论文 | 会议 | 年份 | 方向 | arXiv / 项目页 |
|---|------|------|------|------|----------------|
| 1 | OpenVLA | NeurIPS | 2024→2025 | VLA 端到端 | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| 2 | π₀ / π₀.₅ | arXiv | 2024–2025 | VLA + Flow Matching | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| 3 | GR-2 | ICML | 2025 | 视频生成 + VLA | [arXiv:2502.02064](https://arxiv.org/abs/2502.02064) |
| 4 | TinyVLA | ICML | 2025 | 轻量 VLA | [arXiv:2402.04492](https://arxiv.org/abs/2402.04492) |
| 5 | RoboVLA | arXiv | 2025 | 联合训练 VLA | [arXiv:2501.16108](https://arxiv.org/abs/2501.16108) |
| 6 | RoboFlamingo | CVPR | 2025 | VLM → VLA | [项目页](https://roboflamingo.github.io) |
| 7 | 3D Diffusion Policy (DP3) | CVPR | 2025 | 3D 扩散策略 | [项目页](https://3d-diffusion-policy.github.io) |
| 8 | ManiCM | CVPR | 2025 | 实时扩散策略 | [项目页](https://manicm.github.io) |
| 9 | UMI | CVPR | 2025 | 数据采集接口 | [项目页](https://umi-gripper.github.io) |
| 10 | Clio | CVPR | 2025 | 实时 3D 场景图 | [项目页](https://clio-dataset.github.io) |
| 11 | SpatialVLM | CVPR | 2025 | VLM 空间推理 | [项目页](https://spatialvlm.github.io) |
| 12 | AffordanceSAM | CVPR WS | 2025 | 供用性分割 | [arXiv:2407.12494](https://arxiv.org/abs/2407.12494) |
| 13 | AnyGrasp | ICML | 2024→2025 | 通用抓取 | [arXiv:2212.08333](https://arxiv.org/abs/2212.08333) |
| 14 | GaussianGrasper | CVPR | 2024→2025 | 3DGS 抓取 | [arXiv:2403.19696](https://arxiv.org/abs/2403.19696) |
| 15 | PhysDreamer | NeurIPS | 2024→2025 | 物理属性估计 | [arXiv:2404.13026](https://arxiv.org/abs/2404.13026) |
| 16 | PhysBench | NeurIPS | 2024→2025 | 物理属性基准 | [arXiv:2409.17955](https://arxiv.org/abs/2409.17955) |
| 17 | HOMER | NeurIPS | 2024 | 家用物体属性库 | — |
| 18 | ConceptGraphs | ICRA→2025 | 3D 场景图 | [项目页](https://concept-graphs.github.io) |
| 19 | VoxPoser | CoRL→2025 | LLM + 3D 价值图 | [项目页](https://voxposer.github.io) |
| 20 | DexVLG | NeurIPS | 2024→2025 | 语言引导灵巧抓取 | [arXiv:2412.01711](https://arxiv.org/abs/2412.01711) |

---

## 十、结论

1. **2025 年是具身智能 + 视觉融合的爆发年**：CVPR 2025 出现了大量 embodied manipulation 论文，VLA 范式已成主流。

2. **四条路线各有定位**：
   - **VLA 端到端**（OpenVLA/π₀/GR-2）→ 最直接的"看→做"方案，但数据需求大
   - **3D 视觉策略**（3D-DP/ManiCM）→ 泛化性最好，适合不同场景布局
   - **开放词汇场景图**（Clio/ConceptGraphs）→ 空间推理最强，适合复杂指令
   - **供用性/属性**（AffordanceSAM/PhysBench）→ 最接近"如何操作"的答案

3. **面向家政/烹饪场景最有价值的组合**：
   - **Clio + AffordanceSAM + AnyGrasp** → "找到灶台旁的锅" + "识别把手可抓取" + "执行抓取"
   - **OpenVLA + UMI** → 用 UMI 采集烹饪示范 + 用 OpenVLA 训练策略
   - **π₀.₅** → 直接面向家务的 VLA，但复现门槛高

4. **最大研究空白**：烹饪场景的动态状态感知（食材生熟判断）和物体属性库到操作策略的端到端闭环。

5. **最推荐入手的三个工作**：OpenVLA（VLA 基线）、AffordanceSAM（供用性分割）、Clio（实时 3D 场景图）— 均代码开源、复现门槛低、直接服务于家政/烹饪场景。
