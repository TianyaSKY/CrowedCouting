# 人群计数损失函数设计

本文以当前代码为准，详解主分支 `models/point_moe_loss.py` 的 **`PointMoELoss`**
（点级 Scale-MoE Head）。旧 v4 分支的 `CrowdPointLoss`（`models/loss.py`）见文末附录。

## 1. 目的

数据只有人头中心点、没有可靠框宽高。因此模型不做 IoU/DFL 框回归，而是把 P3 网格 × K 参考点
视为统一候选点集合：分类分支判断是否有人头，偏移分支细化人头中心，Router 决定每个点用哪个
尺度的专家，三者联合优化。

## 2. 候选点与坐标

`MoEPointHead` 只在 P3（stride 8）网格上解码，步长 8、`offset_range=2.0`：

$$
P = (grid + ref_k + \tanh(dx, dy) \cdot 2.0 \cdot 8) \times 8
$$

- `grid`：P3 网格整数坐标；`ref_k`：K 个参考点的相对偏移（K=4 时为 2×2 子格中心）。
- 训练裁剪 640×640 时 Q = 80×80×4 = 25,600 个候选点；Q 随 `--crop-size` 改变。
- 计数直接对全部候选点 `sigmoid(logits)` 求和，不依赖阈值。

## 3. 匹配：Top-K + 匈牙利一对一

对每张图：

1. 按置信度取 top-K 候选（K = max(`match_top_k`=2000, n_gt)）——否则 n_gt×Q 的
   线性指派在密集裁剪（实测单张 1,407 个 GT）上会卡住数分钟；
2. 代价 = `5 × L1(归一化预测点, 归一化 GT 点) − sigmoid(logit)`（坐标主导 + 置信度软约束）；
3. `scipy.optimize.linear_sum_assignment` 匈牙利指派，一对一；
4. 匹配候选为正样本（分类目标 1），其余为负。

与 v4 的“Top-K 锚点 + 距离阈值”不同：这里每个 GT 恰好匹配一个候选点，且允许网络
在密集区域输出“超配”的负候选由分类损失压制，去掉了固定阈值带来的漏检/重复计数边界。

## 4. 总损失

$$
L = L_{cls} + 5L_{point} + 0.05L_{count}
$$

### 4.1 分类损失 `L_cls`

Sigmoid Focal（α=0.25、γ=2.0）：

$$
L_{cls} = \frac{1}{B}\sum_b \frac{\sum_a \alpha_t (1-p_t)^\gamma \text{BCE}(z_a, y_a)}{\max(n_{gt}, 1)}
$$

逐图按 GT 数归一化，再对 batch 平均。

### 4.2 定位损失 `L_point`

仅在匹配对上计算 Smooth L1，坐标归一化到图像尺寸：

$$
L_{point} = \frac{1}{B}\sum_b \frac{\sum_{m} \text{SmoothL1}_{\beta=0.02}(\hat{p}_m, p_m^{gt})}{n_{gt}}
$$

**β=0.02 是关键**：归一化坐标下典型定位误差约 0.005（≈2px/384），若用默认 β=1.0，
所有误差落入二次区，损失被压到 ~1e-5 量级、定位分支几乎没有梯度；β=0.02（≈8px）让
2–8 px 误差走线性/准线性区，保持定位梯度。

### 4.3 计数损失 `L_count`

$$
L_{count} = \frac{1}{B}\sum_b \frac{|\sum_a \sigma(z_a) - n_{gt}|}{n_{gt} + 1}
$$

软计数监督：全图概率和逼近真实人数。相比 v4 已删除的旧计数损失（对全部网格无差别
强梯度），这里权重低（0.05）、分母 +1 平滑，且分类已有 Focal 的难易样本控制。

### 4.4 路由诊断（不参与训练）

`scale_targets` 可用 GT 局部密度（最近邻间距）映射为三专家软目标，
但只用于 matched confusion diagnostic。它不产生 Router 梯度，也不增加 loss 项。
训练不使用 `L_route`、`L_balance`、`L_sharp` 或 Router recall graduation。

## 5. 训练时的路由动态（train_moe.py）

- **D2 warm-up**：默认前 6 epoch 使用 candidate-level Random Drop-1，Router
  不更新。Epoch 1–3 backbone frozen；Epoch 4–6 backbone trainable；Epoch 7+
  Router 和 backbone 都 trainable。warm-up 剩余两个 expert 固定 `0.5/0.5`。
- **Router-active Drop-1**：Router 只决定随机保留下来的两个 expert 之间的
  相对权重，使用 masked softmax；每个 candidate 始终恰好两个非零 gate。
  三个 expert 都先执行，第一版不做 sparse dispatch。
- **双概率**：`route_probabilities` 是未 Drop-1 的完整三专家 softmax，用于
  matched probability/entropy/margin 诊断；`gates` 是实际混合 gate。
- **温度**：Router 启用时重新计数，默认 `T=2.0 → 1.5 → 1.0`，对应
  Router epoch 0/15/30。训练不使用 Gumbel-ST。
- **验证**：每 epoch 计算 full3-soft、deterministic Top-2 和 diagnostic-only
  Top-1；`best_top2.pt` 按 Top-2 weighted normalized MAE 选取。
- **日志**：记录 full3 matched probability mean、entropy、margin、deterministic
  Top-1 usage、Drop-1 frequency、active exposure、masked gate mean，以及定期
  的三个 expert-only MAE。

## 6. 与评估的关系

人数 = `Σσ(logits)`。MoE 分支的计数不经过置信度阈值/NMS，因此 MAE/RMSE 可直接复现；
`best_top2.pt` 按 Top-2 weighted normalized MAE 选取。评估脚本默认从 checkpoint
读取 `crop_size`、`num_references`、保存的温度与 D2 Router 配置。

## 附录：v4 CrowdPointLoss（旧，对照用）

`models/loss.py`，配合 `PointDetect`（P2–P5 四层，imgsz=1024 时 87,040 锚点）：

$$
L = 1.0L_{cls} + 5.0L_{off}
$$

- 匹配：每 GT 取最近 Top-K=3 锚点，阈值 `1.5 × stride`（P2 6 / P3 12 / P4 24 / P5 48 px）；
  多个 GT 匹配同一锚点时各配对分别算偏移损失（不平均、不覆盖），但该锚点仍只能输出
  一个点——极近人头的冲突监督是结构限制。
- `L_cls`：BCE + Focal（γ=2.0、α=0.25），`sum / max(正样本数,1) × 20`。
- `L_off`：正锚点预测/目标偏移的逐元素 L1 均值 × 5。
- 第三槽位恒 0（兼容训练器三槽位）；全局计数损失已移除——早期 34,000 个网格概率求和
  梯度过强且不区分正负样本，干扰分类学习。旧文档中 `0.1 × L_count` 的公式不再适用。
- 评估注意：v4 用检测框数计数、框中心做点匹配，训练日志的 `Box(P/R/mAP)` 因虚拟框
  尺寸（1×1 px vs 标签宽高 1%）IoU 恒不达标，不能用于评价；比较前必须固定置信度阈值、
  点匹配容忍距离、去重规则与 `max_det`。
