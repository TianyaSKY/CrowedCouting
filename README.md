# CrowdCounting

基于 YOLO11 的人群计数项目，包含两个分支：

1. **v4 PointDetect 分支**（早期实现）：在 YOLO11 各尺度上添加逐尺度点检测头。
2. **点级 Scale-MoE Head 分支**（当前推荐）：保留 YOLO11 Backbone+Neck、删除原始
   Detect Head，替换为统一候选点网格上的三专家 MoE 点检测头。

## 目录

- `models/`：模型结构、锚点生成和损失函数。
  - `yolo11_pyramid.py`：`YOLO11Pyramid`，保留 Backbone+Neck、移除 Detect Head，输出 P3/P4/P5。
  - `moe_point_head.py`：`MoEPointHead`，三个专家（局部/中层/大范围感受野）+ 点级 Router。
  - `yolo11_moe_point.py`：`YOLO11MoEPoint`，Backbone+Neck + MoE Point Head 的组合网络。
  - `point_moe_loss.py`：`PointMoELoss`，匈牙利匹配 + Focal 分类 + Smooth L1 + 计数 + 尺度路由监督。
- `scripts/data/`：ShanghaiTech 数据集转换、离线增强与点标注转换。
- `scripts/training/`：训练与恢复训练。
- `scripts/evaluation/`：人数、定位和模型对比评估。
- `scripts/visualization/`：单图预测、结果绘制与可视化对比。
- `scripts/diagnostics/`：不修改数据的匹配逻辑检查工具。
- `runs/`：训练、评估与可视化输出（运行生成）。

## 点级 Scale-MoE Head 分支

结构：

```text
输入图像
   ↓
YOLO11 Backbone
   ↓
YOLO11 PAN-FPN Neck
   ↓
P3(8)  P4(16)  P5(32)   ← 三个尺度只产生同一组 P3 候选点
   ↓       ↓       ↓
Point Router (g3, g4, g5)
   ↓
Expert3(F3) / Expert4(Up(F4)+α4·F3) / Expert5(Up(F5)+α5·F3)
   ↓   α4/α5 可学习横向融合，初始为 0（纯尺度输入）
每个候选点输出 (x, y, confidence)，候选点 = P3 网格 × K 个参考点
```

- 训练早期使用软路由（概率空间加权混合：$p=\sum_i g_i\sigma(z_i)$，再转回 logit），约第 20 epoch 后切换 Top-1 硬路由（Straight-Through）。
- Router 温度从 2.0 线性衰减到 0.5。
- 匹配：候选点与真实点做匈牙利一对一匹配，正候选点学习选择最合适的专家。
- 路由监督：不再强制三专家 1/3 均匀使用，而是用 GT 点最近邻间距估计局部尺度，
  映射为专家软目标（小间距→E3 精细、大间距→E5 大范围），与 Router logits 做交叉熵（`L_route`，权重 0.05）。
  这让 Router 学到数据驱动的尺度语义。

### 准备数据

```bash
# 0. 下载官方数据集（约 166 MB，官方 Dropbox 镜像；也可用 --url 换源）
python -m scripts.data.download_shanghaitech

# 1. 生成合并数据集（现有 prepare_combined 生成 YOLO 虚拟框标签）
python -m scripts.data.prepare_combined

# 2. 将 YOLO 虚拟框标签（cls nx ny w h）转换为纯点标签（nx ny）
python -m scripts.data.prepare_point_labels
```

点标注格式（`datasets/shanghaitech_AB/points/{train,val}/*.txt`，每行）：

```text
0.3125 0.4172
0.4251 0.3928
0.7312 0.6215
```

### 训练

```bash
python -m scripts.training.train_moe \
    --weights yolo11n.pt \
    --data-root datasets/shanghaitech_AB \
    --crop-size 384 \
    --batch-size 128 \
    --epochs 100 \
    --save-dir runs/moe_point
```

推荐超参数（可覆盖）：YOLO 主干学习率 `1e-4`、Head 学习率 `1e-3`、每网格参考点 K=4、
`--hard-route-epoch 20`、`--freeze-epochs 3`、`--route-weight 0.05`。

### 推理

```bash
python -m scripts.visualization.predict_moe \
    --image path/to/image.jpg \
    --checkpoint runs/moe_point/best.pt
```

三个专家用不同颜色绘制：红色=P3 局部细节、绿色=P4 中层上下文、蓝色=P5 大范围上下文。

### 判断专家坍缩

训练日志输出 `route` 项（尺度路由 CE）。路由目标来自 GT 实际尺度分布：
密集场景大量使用 E3 是**正确的尺度专家化**，不是坍缩。若某个专家几乎
从不被使用（其对应尺度区间内几乎没有 GT 样本），应检查 `scale_centers`
阈值是否与数据尺度匹配（默认 10/20/40 px，`models/point_moe_loss.py`
中可调）。

验证使用 letterbox（保持纵横比 + 居中填充）而非直接压成正方形，避免
人为改变人的尺度；日志同时输出 `soft_MAE` 与 `hard_MAE`，硬路由生效后
`best.pt` 按 hard MAE 选取（最终推理即硬路由）。

## v4 PointDetect 分支（旧）

```bash
python -m scripts.training.train_custom_v4
python -m scripts.evaluation.count
python -m scripts.evaluation.localization
python -m scripts.visualization.predict
```

## 运行方式

始终从项目根目录以模块方式执行，避免相对导入和工作目录问题。每个脚本底部保留了
原有的默认路径和参数；需要切换数据集、权重或阈值时，修改对应脚本的 `__main__` 配置即可。
