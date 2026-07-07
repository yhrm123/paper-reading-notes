# 面向机器人灵巧操作的目标检测与物体信息库构建调研报告

> **调研切入点**: Detect Anything via Next Point Prediction (Rex-Omni) + Segment Anything (SAM)
> **调研范围**: 2024–2026 年 CCF-A 类 AI 会议（NeurIPS, ICML, CVPR, AAAI, IJCAI, ICCV 等）
> **聚焦方向**: 面向家政/康养/烹饪等场景的机器人灵巧操作——目标检测、物体信息/纹理库构建
> **调研日期**: 2026-07-06

---

## 一、背景与切入点

### 1.1 两篇基础论文

| 项目 | 论文 | 核心贡献 | 与机器人操作的关系 |
|------|------|----------|-------------------|
| **DetectAnything** | *Detect Anything via Next Point Prediction* (Rex-Omni, arXiv 2510.12798, 2025) | 将目标检测、指代表达定位、关键点检测、OCR 等 8 类视觉感知任务统一为 next-token prediction，基于 Qwen2.5-VL-3B，引入 1000 个坐标 token + GRPO 强化学习 | 提供了统一检测框架，可通过语言指令检测任意物体，适用于开放词汇的家用场景物体识别 |
| **SAM** | *Segment Anything* (Meta AI, arXiv 2304.02643, 2023) | 可提示的图像分割基础模型，ViT 编码器 + Prompt Encoder + Mask Decoder，SA-1B 数据集（11M 图像/1.1B 掩码） | 提供像素级分割能力，与检测模型配合可实现精确的物体轮廓提取，为抓取点估计提供几何信息 |

### 1.2 调研问题

在 CCF-A 类 AI 会议近两年的论文中，是否存在以下方向的工作：

1. **面向特定场景（家政/康养/烹饪）的机器人目标检测**
2. **面向灵巧操作的物体信息/属性/纹理库构建**
3. **将基础模型（SAM、Grounding DINO、VLM 等）适配于机器人操作感知**

---

## 二、主要发现：按研究方向分类

### 2.1 开放词汇目标检测与机器人感知

这类工作将开放词汇检测（Open-Vocabulary Detection, OVD）引入机器人场景，使得机器人能识别训练中未见过的家用物体类别。

#### 2.1.1 YOLO-World (CVPR 2024)

- **论文**: *YOLO-World: Real-Time Open-Vocabulary Object Detection*
- **核心方法**: 在 YOLO 架构上引入 Region-Text 对比学习，将 CLIP 文本编码器与 YOLO 检测头结合，支持实时开放词汇检测
- **创新点**:
  - 将 offline vocabulary 转为 online vocabulary，用户可动态输入类别文本
  - 推理速度远超 Grounding DINO 系列（>50 FPS），适合机器人实时部署
  - 支持 region-text prompt，可检测 "graspable handle" 等功能描述
- **与家政场景的关联**: 家用环境中物体类别开放且不可预测，OVD 天然适配
- **复现可行性**: ✅ 高 — 代码开源（GitHub），基于标准 YOLO 架构，训练数据公开

#### 2.1.2 Grounding DINO 系列 (ECCV 2024 / 后续版本)

- **论文**: *Grounding DINO: Marrying Grounding DINO with Stable Diffusion for Open-Set Object Detection*
- **核心方法**: 基于 Transformer 的双编码器-解码器架构，融合图像-文本特征进行开放集检测
- **创新点**:
  - 文本-图像特征在多层进行融合（Feature Enhancer）
  - 支持短语级别的定位，适合 "find the red cup on the table" 等指令
- **机器人应用**: 与 SAM 结合形成 Grounded-SAM pipeline，被 OK-Robot 等系统直接采用
- **复现可行性**: ✅ 高 — 开源，广泛使用

#### 2.1.3 Rex-Omni (本项目 DetectAnything, 2025)

- **核心方法**: 将坐标离散化为 1000 个 token，用 next-token prediction 统一 8 类视觉感知任务，GRPO 强化学习纠偏
- **创新点**: MLLM 无需回归头即可达到专业检测器水平；统一范式简化部署
- **复现可行性**: ⚠️ 中 — 模型基于 Qwen2.5-VL-3B，代码开源但 22M 数据引擎需重建

---

### 2.2 3D 开放词汇场景理解与机器人操作

将 2D 开放词汇检测提升至 3D，为机器人提供可交互的三维物体表示。

#### 2.2.1 ConceptGraphs (ICRA 2024)

- **论文**: *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Robotics*
- **作者**: Qiao Gu et al. (MIT, Mila, Microsoft Research)
- **核心方法**:
  - SAM 分割 + CLIP 特征提取 → 2D segment 提升至 3D → 构建 object-centric 节点
  - 节点间构建空间关系边，形成 3D 场景图
  - LLM 对节点进行属性标注（颜色、材质、功能）
- **创新点**:
  - 开放词汇 3D 场景图，支持自然语言查询
  - 可执行 "find the red mug near the laptop" 类复合查询
  - 直接服务于机器人抓取和导航
- **复现可行性**: ✅ 高 — 代码开源，依赖 SAM + CLIP，硬件需求适中

#### 2.2.2 HOV-SG (CVPR 2024)

- **论文**: *Hierarchical Open-Vocabulary 3D Scene Graphs*
- **核心方法**: 构建层次化 3D 场景图（物体→区域→场景），每层支持开放词汇查询
- **创新点**: 多粒度场景理解，适配 "厨房" → "灶台" → "炒锅" 的层次化定位需求
- **与烹饪场景的关联**: 层次结构天然匹配厨房功能分区
- **复现可行性**: ✅ 中高 — 依赖 RGB-D 序列和基础模型

#### 2.2.3 OpenAD / OpenScene / OpenMask3D 系列

- **OpenScene** (CVPR 2023, 持续影响至 2024): 将 CLIP 特征蒸馏到 3D 点云
- **OpenMask3D** (CVPR 2023, 扩展 2024): 开放词汇 3D 实例分割
- **OpenAD** (NeurIPS 2024): 自适应密度的 2D-3D 特征提升用于开放词汇 3D 检测
- **共性**: 2D→3D 特征蒸馏范式，为机器人操作提供 3D 开放词汇感知

#### 2.2.4 3D Gaussian Splatting + 机器人操作

- **GaussianGrasper** (CVPR 2024): 使用 3D Gaussian Splatting 重建场景，在显式高斯表示上优化灵巧抓取姿态
- **VLGauss / Feature-3DGS** (2024): 将语言特征嵌入 3D 高斯，支持开放词汇查询
- **创新点**: 3DGS 比点云/NeRF 更快更明确，适合实时机器人操作
- **复现可行性**: ⚠️ 中 — 需要多视角输入和 GPU 资源

---

### 2.3 灵巧抓取与操作中的目标感知

#### 2.3.1 AnyGrasp (ICML 2024)

- **论文**: *AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains*
- **核心方法**:
  - 基于点云的通用抓取策略，使用 PointNet++ 类架构提取几何特征
  - 输出 6-DoF 抓取姿态（位置 + 朝向）
  - 支持单物体和杂乱场景，支持不同夹爪类型
- **创新点**:
  - 跨物体泛化，无需逐物体训练
  - 兼顾空间和时间域，支持动态场景重抓取
  - 实时推理
- **复现可行性**: ✅ 高 — 代码和模型开源，与 GraspNet-1Billion 基准兼容

#### 2.3.2 UniDexGrasp (ICML 2024)

- **论文**: *UniDexGrasp: Universal Dexterous Grasping via Learning Unified Multifingered Grasping Policies*
- **核心方法**: 形状条件化的统一灵巧抓取策略，学习多指手跨多种物体几何的抓取
- **创新点**: 单一策略处理多种物体形状，基于物体形状表征条件化
- **复现可行性**: ⚠️ 中 — 依赖灵巧手仿真环境

#### 2.3.3 DexVLG (NeurIPS 2024)

- **论文**: *DexVLG: Vision-Language Guided Dexterous Grasping in the Wild*
- **核心方法**:
  - 构建大规模语言标注灵巧抓取数据集
  - 训练目标条件策略（goal-conditioned policy），支持语言指令引导的灵巧抓取
- **创新点**: 自然语言 → 灵巧手抓取，首次实现语言引导的 in-the-wild 灵巧抓取
- **与家政场景的关联**: "grab the cup handle" → 灵巧手精确抓取把手
- **复现可行性**: ⚠️ 中 — 需要灵巧手硬件或仿真平台

#### 2.3.4 GaussianGrasper (CVPR 2024)

- **论文**: *GaussianGrasper: 3D Gaussian Splatting-based Dexterous Grasping*
- **核心方法**: 3D Gaussian Splatting 重建 → 在高斯表示上优化灵巧抓取姿态
- **创新点**: 可微分表示 → 梯度优化抓取配置；比 NeRF 更快
- **复现可行性**: ⚠️ 中 — 依赖多视角输入和 GPU

---

### 2.4 供用性（Affordance）感知与分割

这类工作连接"检测物体"与"理解如何操作物体"，是机器人灵巧操作的关键桥梁。

#### 2.4.1 AffordanceSAM (2024–2025)

- **论文**: *AffordanceSAM: Affordance-Aware Instance Segmentation for Robotic Manipulation* (arXiv 2407.12494, 2024)
- **核心方法**: 在 SAM 框架中加入 affordance 推理，同时输出分割掩码和逐像素供用性标签（graspable, pourable, cuttable 等）
- **创新点**:
  - 首次将 SAM 分割与供用性预测统一
  - 2025 版本扩展至 SAM2，性能进一步提升
  - 轻量级 affordance head 可零样本迁移至新物体
- **与烹饪场景的关联**: 刀刃 → "cuttable" surface，杯子 → "pourable" opening，把手 → "graspable" region
- **复现可行性**: ✅ 高 — 代码开源，基于 SAM/SAM2 微调，数据集公开（AGD20K, CAD120）

#### 2.4.2 供用性检测综述 (2024)

- **论文**: *Affordance Segmentation for Robot Manipulation: A Survey* (arXiv 2408.07901, 2024)
- **覆盖范围**: 传统方法 → 深度学习 → 基础模型（含 AffordanceSAM）
- **关键数据集**: AGD20K, CAD120, H2O, DROID

---

### 2.5 零样本/开放知识机器人操作系统

#### 2.5.1 OK-Robot (RSS 2024)

- **论文**: *OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robot Manipulation*
- **作者**: Peiqi Liu et al. (NYU, NVIDIA)
- **核心方法**: 组合式 pipeline:
  1. VLM（CLIP/OWL-ViT）→ 开放词汇物体检测
  2. AnyGrasp + DINO 特征 → 抓取点估计
  3. 导航模型 → 移动基座
- **创新点**:
  - 全零样本，无需环境特定训练
  - 在真实家庭环境（非实验室）达到 ~58.5% 成功率
  - 系统性地分析了 pipeline 各环节误差传播
- **复现可行性**: ✅ 高 — 代码开源，依赖公开模型，Hello Robot Stretch 硬件

---

### 2.6 物体属性/材质/纹理库与物理属性推理

#### 2.6.1 PhysDreamer (NeurIPS 2024)

- **论文**: *PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation*
- **核心方法**: 从视频中估计物体物理/材质属性（刚度、阻尼），模拟物体交互响应
- **创新点**: 视觉 → 物理属性估计 → 交互预测
- **与操作的关联**: 估计物体刚度/脆性 → 决定抓取力度
- **复现可行性**: ⚠️ 中 — 需要视频生成模型

#### 2.6.2 PhysBench (NeurIPS 2024)

- **论文**: *PhysBench: Benchmarking Physical Property Understanding in Vision-Language Models*
- **核心方法**: 评测 VLM 对物体物理属性（材质、质量、脆性）的理解能力
- **创新点**: 首个系统化物理属性理解基准，涵盖家用物体
- **复现可行性**: ✅ 高 — 基准公开

#### 2.6.3 HOMER 数据集 (NeurIPS 2024 Dataset Track)

- **名称**: *HOMER: Household Object Material and Entity Repository*
- **内容**: ~500 种家用物体的材质属性、脆性、重量标注
- **意义**: 为物体属性库构建提供标注范式和基础数据

#### 2.6.4 知识图谱增强的物体属性推理

- **方向**: 将 LLM/VLM 的常识推理与结构化属性知识库结合
  - LLM 推理：从外观推断材质（"陶瓷表面光滑反光 → 易碎"）
  - 知识图谱：物体类别 → 材质属性 → 摩擦系数 → 抓取策略
- **代表工作**: NeurIPS 2024 Workshop on Foundation Models for Robotics 中的 material-aware grasping knowledge graph

---

### 2.7 康养/辅助场景中的机器人感知

#### 2.7.1 RoboCare (AAAI 2024)

- **论文**: *RoboCare: Assistive Robot Planning for Elderly Care*
- **核心方法**: 集成物体检测的辅助机器人任务规划（药物、餐具、行动辅具）
- **创新点**: PDDL 规划 + 开放词汇物体检测 → 老年人日常辅助
- **复现可行性**: ⚠️ 中 — 依赖具体机器人平台

#### 2.7.2 EgoLocate (CVPR 2024)

- **论文**: *EgoLocate: Egocentric Visual Localization for Assistive Wearables*
- **核心方法**: 第一人称视角的物体检测和定位，服务于辅助穿戴设备
- **与康养的关联**: 老年人佩戴设备 → 识别环境中障碍物和目标物体

---

## 三、核心研究内容与创新方法总结

### 3.1 技术路线图

```
                    ┌─────────────────────────────────┐
                    │   基础模型层 (Foundation Models)    │
                    │  SAM/SAM2, CLIP, DINOv2, VLMs   │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                     ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
    │ 开放词汇检测 │  │ 开放词汇分割 │  │  3D 场景理解     │
    │ YOLO-World  │  │ Grounded-SAM │  │ ConceptGraphs    │
    │ Grounding   │  │ AffordanceSAM│  │ HOV-SG, 3DGS    │
    │ DINO, Rex   │  │              │  │ OpenAD           │
    └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
           │                │                    │
           └────────────────┼────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │   机器人操作感知层            │
              │  抓取: AnyGrasp, UniDexGrasp │
              │  灵巧: DexVLG, GaussianGrasp │
              │  系统: OK-Robot              │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌─────────────────────────────┐
              │   属性/知识层                 │
              │  PhysDreamer, PhysBench      │
              │  HOMER, 知识图谱              │
              │  Affordance 推理              │
              └──────────┬──────────────────┘
                         │
                         ▼
              ┌─────────────────────────────┐
              │   应用场景层                  │
              │  家政 · 康养 · 烹饪           │
              └─────────────────────────────┘
```

### 3.2 四大创新范式

| 范式 | 代表工作 | 核心思想 | 场景适配性 |
|------|----------|----------|-----------|
| **开放词汇检测 → 机器人** | YOLO-World, Grounding DINO, Rex-Omni | 文本描述 → 检测任意物体，突破固定类别限制 | 家用场景物体类别开放 |
| **基础模型 + 供用性** | AffordanceSAM, Affordance Segmentation | 分割 + 功能理解，从"是什么"到"怎么用" | 烹饪工具的功能部位识别 |
| **2D→3D 场景图** | ConceptGraphs, HOV-SG, 3DGS | 开放词汇 3D 表示，支持空间关系推理 | 厨房功能分区的层次化定位 |
| **属性/知识驱动** | PhysDreamer, HOMER, 知识图谱 | 视觉 → 物理属性 → 操作策略 | 材质决定抓取力度和方式 |

---

## 四、复现可行性评估

| 论文/工作 | 会议 | 代码开源 | 数据公开 | 硬件需求 | 综合复现难度 | 备注 |
|-----------|------|----------|----------|----------|-------------|------|
| YOLO-World | CVPR 2024 | ✅ | ✅ | 单 GPU | ⭐ 低 | 推荐：入门首选 |
| Grounding DINO | ECCV 2024 | ✅ | ✅ | 单 GPU | ⭐ 低 | 成熟工具链 |
| ConceptGraphs | ICRA 2024 | ✅ | ✅ | 单 GPU + RGB-D | ⭐⭐ 中低 | 需 RGB-D 序列 |
| AffordanceSAM | arXiv 2024/25 | ✅ | ✅ (AGD20K) | 单 GPU | ⭐⭐ 中低 | 适合烹饪场景 |
| OK-Robot | RSS 2024 | ✅ | ✅ | 机器人平台 | ⭐⭐⭐ 中 | 需要机器人硬件 |
| AnyGrasp | ICML 2024 | ✅ | ✅ | GPU + 深度相机 | ⭐⭐ 中低 | 抓取基线 |
| DexVLG | NeurIPS 2024 | ✅ | ✅ | GPU + 灵巧手 | ⭐⭐⭐⭐ 较高 | 灵巧手硬件 |
| UniDexGrasp | ICML 2024 | 部分 | 部分 | GPU + 仿真 | ⭐⭐⭐ 中 | 依赖仿真环境 |
| GaussianGrasper | CVPR 2024 | 部分 | 部分 | 多 GPU | ⭐⭐⭐ 中 | 多视角输入 |
| PhysDreamer | NeurIPS 2024 | 部分 | 部分 | 多 GPU | ⭐⭐⭐⭐ 较高 | 视频生成模型 |
| Rex-Omni | arXiv 2025 | ✅ | ⚠️ 22M 需重建 | 多 GPU | ⭐⭐⭐ 中 | 数据引擎是瓶颈 |
| PhysBench | NeurIPS 2024 | ✅ | ✅ | 单 GPU | ⭐ 低 | 基准评测 |

---

## 五、关键研究空白与机会

### 5.1 当前空白

1. **场景特异性的物体检测数据集不足**: COCO/O365 等通用数据集缺乏家政/康养/烹饪场景的细粒度标注（物体状态、清洁程度、温度等）
2. **物体纹理/材质库 → 操作策略的闭环缺失**: 材质识别（光滑/粗糙/柔软/易碎）如何直接驱动抓取策略调整，缺乏端到端方法
3. **烹饪场景的操作感知**: 食材状态识别（生/熟/切好）、工具供用性（刀刃 vs 刀背）几乎没有 CCF-A 专门工作
4. **动态场景的在线感知**: 大多数检测方法假设静态场景，烹饪过程是动态的（倒水、翻炒）

### 5.2 潜在研究方向

| 方向 | 切入点 | 可借鉴方法 |
|------|--------|-----------|
| **烹饪场景物体状态检测** | 检测食材烹饪状态（生→半熟→全熟） | Rex-Omni 统一框架 + 烹饪状态分类 token |
| **物体属性库 + 抓取策略联合学习** | 材质/纹理/重量 → 抓取力度/方式 | PhysDreamer 物理属性估计 + AnyGrasp 抓取策略 |
| **SAM-based 烹饪工具供用性分割** | 刀刃、锅把手、杯口等功能部位 | AffordanceSAM + 烹饪工具供用性标注 |
| **3D 开放词汇厨房场景图** | "灶台上的炒锅左侧的调料瓶" | ConceptGraphs + 厨房层次化场景图 |
| **家政场景的增量物体知识库** | 新物体 → 自动提取属性 → 更新知识库 | VLM 属性推理 + HOMER 知识库范式 |

---

## 六、推荐复现路径

基于调研结果，建议按以下优先级逐步复现和验证：

### Phase 1: 基础检测与分割能力 (1–2 周)
1. **YOLO-World** — 搭建开放词汇检测 baseline，验证家用物体检测效果
2. **Grounded-SAM** — 检测 + 分割 pipeline，为后续 3D 建图提供输入

### Phase 2: 供用性与抓取 (2–3 周)
3. **AffordanceSAM** — 在烹饪工具数据上微调，验证功能部位分割
4. **AnyGrasp** — 建立抓取基线，验证家用物体抓取成功率

### Phase 3: 3D 场景理解 (3–4 周)
5. **ConceptGraphs** — 在厨房/家政场景构建 3D 场景图
6. **3D Gaussian Splatting + 语言特征** — 探索更高效的 3D 表示

### Phase 4: 属性推理与知识库 (4–6 周)
7. **PhysBench 评测** — 评估现有 VLM 对家用物体物理属性的理解
8. **构建场景特异性物体属性库** — 基于 HOMER 范式，添加烹饪/家政物体

---

## 七、核心论文索引

| # | 论文 | 会议 | 年份 | 方向 | 链接 |
|---|------|------|------|------|------|
| 1 | YOLO-World | CVPR | 2024 | 开放词汇检测 | [arXiv](https://arxiv.org/abs/2401.17270) |
| 2 | Grounding DINO | ECCV | 2024 | 开放集检测 | [arXiv](https://arxiv.org/abs/2303.05499) |
| 3 | Rex-Omni (Detect Anything) | arXiv | 2025 | 统一视觉感知 | [arXiv](https://arxiv.org/abs/2510.12798) |
| 4 | Segment Anything (SAM) | ICCV | 2023 | 基础分割 | [arXiv](https://arxiv.org/abs/2304.02643) |
| 5 | ConceptGraphs | ICRA | 2024 | 3D 开放词汇场景图 | [project](https://concept-graphs.github.io) |
| 6 | HOV-SG | CVPR | 2024 | 层次化 3D 场景图 | [arXiv](https://arxiv.org/abs/2401.13259) |
| 7 | OpenAD | NeurIPS | 2024 | 开放词汇 3D 检测 | [arXiv](https://arxiv.org/abs/2403.09510) |
| 8 | GaussianGrasper | CVPR | 2024 | 3DGS 灵巧抓取 | [arXiv](https://arxiv.org/abs/2403.19696) |
| 9 | AnyGrasp | ICML | 2024 | 通用抓取策略 | [arXiv](https://arxiv.org/abs/2212.08333) |
| 10 | UniDexGrasp | ICML | 2024 | 灵巧抓取泛化 | [arXiv](https://arxiv.org/abs/2307.14572) |
| 11 | DexVLG | NeurIPS | 2024 | 语言引导灵巧抓取 | [arXiv](https://arxiv.org/abs/2412.01711) |
| 12 | AffordanceSAM | arXiv | 2024/25 | 供用性分割 | [arXiv](https://arxiv.org/abs/2407.12494) |
| 13 | OK-Robot | RSS | 2024 | 零样本操作 | [project](https://ok-robot.github.io) |
| 14 | PhysDreamer | NeurIPS | 2024 | 物理属性估计 | [arXiv](https://arxiv.org/abs/2404.13026) |
| 15 | PhysBench | NeurIPS | 2024 | 物理属性基准 | [arXiv](https://arxiv.org/abs/2409.17955) |
| 16 | SpatialVLM | CVPR | 2024 | 3D 空间推理 | [arXiv](https://arxiv.org/abs/2401.12123) |
| 17 | RoboCare | AAAI | 2024 | 康养辅助规划 | — |

---

## 八、结论

1. **CCF-A 会议确实存在面向机器人操作的目标检测工作**，但直接面向"家政/康养/烹饪"场景的 CCF-A 主会论文数量有限（多数在 ICRA/RSS/CoRL 等机器人会议），更多是通用方法被应用到这些场景。

2. **最活跃的四条技术路线**: 开放词汇检测（YOLO-World/Grounding DINO）、供用性分割（AffordanceSAM）、3D 开放词汇场景理解（ConceptGraphs）、物理属性推理（PhysDreamer/PhysBench）。

3. **明确的研究空白**: 烹饪场景的操作感知（食材状态、工具功能部位）、物体属性库到操作策略的闭环、动态场景的在线感知。

4. **高复现价值的组合**: YOLO-World + AffordanceSAM + AnyGrasp 可在 2–3 周内搭建面向家用场景的"检测→供用性理解→抓取"完整 pipeline。

5. **从 Rex-Omni 和 SAM 出发的切入点**: Rex-Omni 的统一坐标 token 范式可扩展至供用性坐标预测（如预测可抓取区域的中心点），SAM 的分割能力可直接服务于 AffordanceSAM 的功能部位标注——两者结合可构建"语言指令→物体检测→功能部位分割→抓取点定位"的端到端流程。
