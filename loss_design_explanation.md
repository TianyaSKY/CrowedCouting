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
- 训练裁剪 384×384 时 Q = 48×48×4 = 9,216 个候选点；Q 随 `--crop-size` 改变。
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
L = L_{cls} + 5L_{point} + 0.05L_{count} + 0.05L_{route}
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

### 4.4 尺度路由损失 `L_route`

用 GT 局部密度（最近邻间距）监督 Router 的尺度选择，替代早期强制 1/3 均匀使用的
balance loss：

1. **局部尺度**：第 `knn_k`（默认 1）近邻间距 `d`；
2. **软目标**：对数尺度空间高斯映射
   $$
   soft\_target_j = \text{softmax}_j\Big(-\tfrac12 \big(\tfrac{\log_2 d - \log_2 c_j}{0.6}\big)^2\Big),
   \quad c = (10, 20, 40)\ \text{px}
   $$
   间距小（密集）→ E3 精细；间距大（稀疏/大目标）→ E5 大范围；
3. **macro 类别平衡**：按软目标 argmax 的类别把匹配点分组，组内软目标 CE 取均值后
   再对三组取平均。GT 目标分布约 E0 60% / E1 27% / E2 13%，逐点 CE 会让 E2 永远学不动；
   分组平均让三类对 Router 同等重要，但**不强制**路由比例 33/33/33。

该监督只要求 GT 点数 ≥ knn_k+1（默认 ≥2），否则该图跳过 route 项。

## 5. 训练时的路由动态（train_moe.py）

- **温度**：软阶段 `init_temperature=2.0` →（`--router-grad-epoch`=15 处 1.3）→
  `--soft-temp-floor`=1.0（`--temp-floor-epoch`=30）；硬阶段 1.0 → `--min-temperature`=0.5
  （`--hard-temp-epochs`=20）。高温下 gate 接近均匀，避免训练初期路由坍缩。
- **Router 梯度隔离**：epoch < 15 时 `router_grad=False`，混合用 gate detach——
  cls/point/count 不向 Router 回传，Router 只由 `L_route` 训练。否则 winner-take-all
  正反馈（质量好的专家被选中更多 → 更多任务梯度 → 更强）会让少数专家饿死。
- **软混合在概率空间**：`p = Σ gᵢ·σ(zᵢ)` 再转回 logit，避免 logits 线性混合时专家
  置信度相互抵消；硬路由 one-hot 时退化为 `logit(σ(z_j)) = z_j` 的单专家 logit。
- **硬路由切换由毕业条件触发**（非固定 epoch）：验证集路由混淆矩阵 E0/E1/E2 行 recall
  连续 `--graduate-stable-epochs`=3 轮 ≥ 0.60/0.40/0.30 且 macro recall ≥ 0.50 才切；
  切硬路由时 best_mae 重置。日志中 `target`（GT 尺度分布）vs `gate`（预测路由分布）
  与硬路由使用率用于判断路由是否学偏。

## 6. 与评估的关系

评估固定 `temperature=0.5`，同时报告 soft/hard MAE；人数 = `Σσ(logits)`。MoE 分支的
计数不经过置信度阈值/NMS，因此 MAE/RMSE 可直接复现；`best.pt` 在软阶段按 soft MAE、
硬阶段按 hard MAE 选取（最终推理即硬路由）。若要比较不同 checkpoint，请固定
`--crop-size`（验证 letterbox 到该尺寸）与 `--temperature`。

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
