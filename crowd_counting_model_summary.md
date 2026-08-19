# 人群计数模型整理报告

本文以当前代码为准。主分支为**点级 Scale-MoE Head**（`models/yolo11_moe_point.py`）。

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
    E --> F[PointMoELoss: 匹配 + cls/point/count]
    E --> G[点数 = Σ sigmoid(logits) 直接计数]
    G --> H[soft_MAE / hard_MAE 评估]
```

## 2. 当前模型：YOLO11 + 点级 Scale-MoE Head

### 2.1 YOLO11Pyramid（`models/yolo11_pyramid.py`）

- 加载 `YOLO(weights)`（默认 `yolo11m.pt`），删除末尾 Detect Head，保留 Backbone + Neck。
- 输出层索引与步长**从 Detect Head 的 `f` / `stride` 字段动态读取**（yolo11m 为第 16/19/22 层，
  stride 8/16/32），不硬编码，可平滑换 yolo11n/s/l 等尺寸。
- 前向用 `save_indices` 缓存中间层，返回 `[P3, P4, P5]`。

### 2.2 MoEPointHead（`models/moe_point_head.py`）

输入 P3/P4/P5，输出统一候选点集合（**P3 网格 × K 个参考点**，K=`num_references`∈{1,4,9}，
默认 4；网格 384²/8²=2304，总候选 Q=2304×4=9216）：

- 投影：`proj3/4/5`（1×1 Conv + GroupNorm + SiLU）把三尺度压到 `hidden_channels=256`；
  f4/f5 双线性上采样到 P3 尺寸。
- **专家**：三个 `PointExpert` 只在 P3 网格上预测，输入为各自尺度特征
  `E3(F3)`、`E4(Up(F4)+α4·F3)`、`E5(Up(F5)+α5·F3)`；α4/α5 为可学习横向融合系数、
  **初始 0**（专家先吃纯尺度输入，防止共享 P3 细节后退化成近似副本）。
  每个专家输出 `K×(confidence, dx, dy)`；confidence 偏置按先验 0.01 初始化。
- **Router**：输入 `Concat(F3, Up(F4), Up(F5))`，输出 `[K, 3]` 原始路由 logits。
  `route_probabilities = softmax(route_logits / T)` 始终保留完整三专家概率，
  不受 Drop-1 或验证模式影响。
- **D2 路由**：
  - 训练每个 candidate 随机 Drop-1；warm-up 时剩余两个固定 `0.5/0.5`，
    Router 启用后在 masked logits 上做 softmax。
  - 验证 `full3_soft` 使用完整 softmax；`top2` 保留两个最大概率并重新归一化；
    `top1` 为 deterministic argmax，仅作 diagnostic。
  - `expert_only` 可强制只启用一个 expert 进行 starvation 诊断。
- **混合**：在概率空间混合 `p = Σ gᵢ·σ(zᵢ)` 再转回 logit，避免 logits 线性混合
  时专家置信度相互抵消。Router 只接收 task-only loss 梯度。
- **解码**：参考点 = (网格 + K 个相对偏移) × 8；预测点 = `参考点 +
  tanh(dx,dy)·offset_range(2.0)·8`。
- 输出：`logits [B,Q]`、`points [B,Q,2]`、`gates [B,Q,3]`、`base_points`、
  `route_logits`、完整 `route_probabilities`、`dropped_expert`、
  `active_expert_mask`。

## 3. PointMoELoss（`models/point_moe_loss.py`）

总损失：

$$
L = L_{cls} + 5L_{point} + 0.05L_{count}
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
- 不使用 `L_route`、`L_balance`、`L_sharp` 或 Router recall graduation。

### 3.3 尺度路由诊断（不参与训练）

`scale_targets` 仍可用 GT 点第 `knn_k`（默认 1，即最近邻）间距估计局部尺度，
在对数尺度空间映射为三专家软目标。它只用于 matched confusion diagnostic，
不产生 Router 梯度，也不改变 task-only loss。

`scale_centers = (10, 20, 40) px`，`scale_sigma_octaves = 0.6`；日志记录完整
`route_probabilities` 的 matched probability mean、entropy、margin、deterministic
Top-1 usage。每 5 epoch 可额外运行三个 expert-only MAE。

## 4. 训练与评估口径（train_moe.py）

- 训练：AdamW 两组学习率（YOLO 主干 1e-4 / MoE Head 1e-3），前 3 epoch 冻结主干；
  前 6 epoch Router 冻结，全部阶段使用 candidate-level Random Drop-1。
- Epoch 1–3 是 backbone frozen + Router frozen；Epoch 4–6 backbone trainable
  + Router frozen；Epoch 7+ 两者都 trainable。Router 启用后只在两个活跃
  expert 间使用 masked softmax。
- 温度只在 Router 启用时重新计数，默认 `2.0 → 1.5 → 1.0`（Router epoch
  0/15/30）；不使用 Gumbel、Router loss、balance loss 或 sharp loss。
- 评估每 epoch 同时计算 `full3-soft`、deterministic `Top-2` 和 diagnostic-only
  `Top-1`。人数 = 全部候选点 `logits.sigmoid()` 之和；`best_top2.pt` 按
  weighted normalized Top-2 MAE 选取。
- checkpoint 记录 `router_training_mode="task_only_drop1_soft_top2"`、
  `expert_dropout="candidate_drop1"`、`active_experts=2`、
  `router_start_epoch=6` 和 `route_supervision=False`。
- 验证用 letterbox（保持纵横比 + 居中填充 114）而非压成正方形，评估脚本默认读取
  checkpoint 记录的 `crop_size`、`num_references`、temperature 和 Router 配置。


