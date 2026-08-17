# 人群计数模型整理报告

本文以当前代码为准。主分支为**点级 Scale-MoE Head**（`models/yolo11_moe_point.py`），
旧 v4 PointDetect 分支（`models/crowd_model.py`）见文末附录，仅作对照参考。

## 1. 系统概览

所有分支都只学习人头中心点，不回归边界框宽高。ShanghaiTech 的 .mat 点标注先转换为
YOLO 格式的**虚拟框**（`prepare_combined.py`，`0 nx ny 0.010000 0.010000`）以复用
ultralytics 的数据约定，再用 `prepare_point_labels.py` 转成纯点标签 `nx ny`；
`PointDataset` 直接读取点标签。

```mermaid
graph TD
    A[.mat 点标注] -->|prepare_combined.py| B[YOLO 虚拟框: 中心 + 0.01 宽高]
    B -->|prepare_point_labels.py| C[points/: nx ny 纯点标签]
    C --> D[YOLO11Pyramid: Backbone+Neck -> P3/P4/P5]
    D --> E[MoEPointHead: 三专家 + Router]
    E --> F[PointMoELoss: 匹配 + cls/point/count/route]
    E --> G[点数 = Σ sigmoid(logits) 直接计数]
    G --> H[soft_MAE / hard_MAE 评估]
```

## 2. 当前模型：YOLO11 + 点级 Scale-MoE Head

### 2.1 YOLO11Pyramid（`models/yolo11_pyramid.py`）

- 加载 `YOLO(weights)`（默认 `yolo11n.pt`），删除末尾 Detect Head，保留 Backbone + Neck。
- 输出层索引与步长**从 Detect Head 的 `f` / `stride` 字段动态读取**（yolo11n 为第 16/19/22 层，
  stride 8/16/32），不硬编码，可平滑换 yolo11s/m/l 等尺寸。
- 前向用 `save_indices` 缓存中间层，返回 `[P3, P4, P5]`。

### 2.2 MoEPointHead（`models/moe_point_head.py`）

输入 P3/P4/P5，输出统一候选点集合（**P3 网格 × K 个参考点**，K=`num_references`∈{1,4,9}，
默认 4；网格 384²/8²=2304，总候选 Q=2304×4=9216）：

- 投影：`proj3/4/5`（1×1 Conv + GroupNorm + SiLU）把三尺度压到 `hidden_channels=128`；
  f4/f5 双线性上采样到 P3 尺寸。
- **专家**：三个 `PointExpert` 只在 P3 网格上预测，输入为各自尺度特征
  `E3(F3)`、`E4(Up(F4)+α4·F3)`、`E5(Up(F5)+α5·F3)`；α4/α5 为可学习横向融合系数、
  **初始 0**（专家先吃纯尺度输入，防止共享 P3 细节后退化成近似副本）。
  每个专家输出 `K×(confidence, dx, dy)`；confidence 偏置按先验 0.01 初始化。
- **Router**：输入 `Concat(F3, Up(F4), Up(F5))`，输出 `[K, 3]` 原始路由 logits
  （初始化全 0 → softmax 后三专家等权）。路由 logits 除以温度后 softmax 得到 gate。
- **软/硬路由**：
  - 软：在**概率空间**混合 `p = Σ gᵢ·σ(zᵢ)` 再转回 logit，避免 logits 线性混合时
    专家置信度相互抵消；`g = softmax(route_logits / T)`。
  - 硬：Top-1 one-hot；训练时走 Straight-Through（前向硬、梯度经软路由回传），
    推理时纯硬路由。
  - `router_grad=False`（训练早期）时混合用 gate 被 detach，cls/point/count 不向
    Router 回传，Router 只由 `L_route` 训练。
- **解码**：参考点 = (网格 + K 个相对偏移) × 8；预测点 = 参考点 + `tanh(dx,dy)·offset_range(2.0)·8`。
- 输出：`logits [B,Q]`、`points [B,Q,2]`、`gates [B,Q,3]`、`base_points`、`route_logits`。

## 3. PointMoELoss（`models/point_moe_loss.py`）

总损失：

$$
L = L_{cls} + 5L_{point} + 0.05L_{count} + 0.05L_{route}
$$

### 3.1 匹配（匈牙利一对一）

网络输出全部 Q 个候选点 → 按置信度取 top-K 作为匹配候选（K = max(2000, n_gt)，
避免 n_gt×Q 的线性指派在密集裁剪上卡死）→ 代价矩阵

$$
cost = 5 \cdot L_1(\hat{p}_{norm}, p_{gt,norm}) - \sigma(logit)
$$

→ `scipy.optimize.linear_sum_assignment` 匈牙利指派。匹配上的候选为正样本，其余为负。

### 3.2 子损失

- **L_cls**：Sigmoid Focal（α=0.25、γ=2.0），逐图 `sum / max(n_gt, 1)`，再对 batch 取平均。
- **L_point**：仅在匹配对上的 Smooth L1（**β=0.02**，归一化坐标下约对应 8 px，让 2–8 px
  误差走线性/准线性区，避免 β=1.0 时定位分支梯度被压到 ~1e-5 量级），`sum / n_gt`。
- **L_count**：`|Σσ(logits) − n_gt| / (n_gt + 1)`，软计数监督。
- **L_route**：尺度路由监督，替代强制 1/3 均匀使用的 balance loss（见下）。

### 3.3 尺度路由监督（L_route）

对每个匹配正样本，用 GT 点第 `knn_k`（默认 1，即最近邻）间距估计局部尺度，在
**对数尺度空间**做高斯映射为三专家软目标：

$$
log\_scale = \log_2(clamp(d_{knn}, 1)), \qquad
soft\_target = softmax\Big(-\tfrac12 \big(\tfrac{\log_2 d - \log_2 c_j}{0.6}\big)^2\Big)_j
$$

`scale_centers = (10, 20, 40) px`，`scale_sigma_octaves = 0.6`：间距小（密集人群）→ E3 精细，
间距大（稀疏/大目标）→ E5 大范围。路由损失收集一个 batch 的所有匹配点，
按硬目标类别分别求均值后再做 macro 平均；三个类别对 Router 训练同等重要，
但不强制最终路由比例 33/33/33。

返回 `(total, {cls, point, count, route, gate_target, gate_target_hist, gate_target_count})`；
`gate_target` 为 GT 尺度目标 argmax 分布，另两个字段用于按真实 support 聚合训练日志。

## 4. 训练与评估口径（train_moe.py）

- 训练：AdamW 两组学习率（YOLO 主干 1e-4 / MoE Head 1e-3），前 3 epoch 冻结主干；
  温度按 schedule 变化；硬路由切换由 **Router 毕业条件**触发（验证集混淆矩阵 E0/E1/E2 行 recall
  连续 3 轮 ≥ 0.60/0.40/0.30 且 macro recall ≥ 0.50），可用 `--force-hard-epoch` 覆盖。
  `router_grad=False` 的 warm-up 使用 uniform floor（默认 0.3）保护少数专家。
- 评估：soft 使用当前 epoch 的训练温度，hard 使用 0.5；**人数 = 全部候选点
  `logits.sigmoid()` 之和**（无阈值、无去重）。`best_soft.pt` / `best_hard.pt` 分别按对应
  phase 的 weighted normalized MAE 选取；Router 未毕业时不生成 `best_hard.pt`。
- 验证用 letterbox（保持纵横比 + 居中填充 114）而非压成正方形，评估脚本默认读取 checkpoint
  记录的 `crop_size`、`num_references`、temperature schedule 和 Router 配置。

## 5. 附录：v4 PointDetect 分支（旧，对照用）

### 5.1 网络（`models/crowd_model.py` + `models/yolo11-crowd.yaml`）

- YAML 为 YOLO11n 风格 Backbone（0–10 层）+ Neck，原生 `Detect` 头（P2/4、P3/8、P4/16、
  P5/32）在 `CrowdCountingModel.__init__` 中被替换为 `PointDetect`。
- 训练器（`train_custom_v3/v4.py`）会尝试从本地 `yolo26n.pt` 载入 Backbone 0–10 层
  形状一致的参数（`load_from_pretrained`）。
- `PointDetect`（`models/modules.py`）：每尺度每网格输出 1 个分类 logit + 2 个未约束偏移
  `(dx, dy)`；推理坐标 `P = (A + (dx,dy)) × s`（A 为带 0.5 偏移的网格中心）。为兼容
  ultralytics 后处理，点被包装成 1×1 px 虚拟框。

### 5.2 CrowdPointLoss（`models/loss.py`）

$$
L = 1.0L_{cls} + 5.0L_{off}
$$

- 匹配：每 GT 点取欧氏距离最近的 Top-K 锚点（K=3），阈值 `1.5 × stride`（P2 6px、P3 12px、
  P4 24px、P5 48px）内的为正样本；同一锚点被多个 GT 匹配时各配对分别参与偏移损失
  （不取平均），但一个锚点仍只能输出一个点——极近人头的结构性限制。
- `L_cls`：BCE + Focal（γ=2.0、α=0.25），按正样本数归一化 ×20。
- `L_off`：正锚点上预测与目标偏移的逐元素 L1 均值 ×5。第三槽位恒 0（兼容训练器）。
- 已移除全局计数损失：早期全图概率求和梯度过大且不区分正负样本，干扰分类学习。

### 5.3 v4 数据规模与评估边界

- `train_custom_v4.py` 固定 `imgsz=1024`，四层锚点共 87,040：P2 65,536 + P3 16,384 +
  P4 4,096 + P5 1,024（640 输入/34,000 锚点的旧配置不适用）。
- v4 评估以**点级指标**为准：MAE/RMSE（`count.py`，扫 conf 阈值）、给定像素容忍距离下的
  一对一匹配 P/R/F1（`localization.py`，默认 conf=0.22、容忍 15 px）、逐图明细
  （`detailed.py`）。训练日志的标准 YOLO `Box(P/R/mAP)` 因虚拟框尺寸（1×1 px vs 标签
  宽高 1%）IoU 恒不达标，不能作为结论；置信度阈值、容忍距离、去重规则与 `max_det`
  必须固定并记录后才可比较。
