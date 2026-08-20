# CrowdCounting

基于 YOLO11 的人群计数项目。模型不回归边界框，只预测人头中心点，支持多数据集。

当前唯一模型是 **native_multiscale**：

```text
P3 -> E0(K=1)  -> 6400 candidates on 640x640
P4 -> E1(K=4)  -> 6400 candidates on 640x640
P5 -> E2(K=16) -> 6400 candidates on 640x640
                 |
                 +-> concatenate -> global Hungarian matching
```

三个 Expert 使用各自的特征层和 reference grid。参考点实际间距约为 8px，坐标偏移能力统一为约 ±16px。模型不包含 Router、Drop-1、Top-2 或尺度标签监督。

## 环境

```bash
pip install -r requirements.txt
# GPU 机器安装匹配 CUDA 的 PyTorch wheel
```

始终从项目根目录以模块方式运行：

```bash
python -m scripts.training.train_moe --help
```

## 目录

- `models/yolo11_pyramid.py`：YOLO11 Backbone+Neck，输出 P3/P4/P5。
- `models/moe_point_head.py`：native P3/P4/P5 Expert head。
- `models/yolo11_moe_point.py`：YOLO11 与 native point head 组合网络。
- `models/point_moe_loss.py`：独立 warmup matching、全局 competitive matching、Focal/Smooth-L1/count loss。
- `scripts/training/train_moe.py`：单数据集训练。
- `scripts/training/train_all.py`：多数据集联合训练与逐数据集验证。
- `scripts/evaluation/evaluate_datasets.py`：跨数据集评估。
- `scripts/visualization/`：单图、批量和 TensorBoard 验证图推理。

旧 D2 Router checkpoint、旧 Top-2/Top-1 评估模式和旧 Router CLI 已删除，不能继续加载。

## 数据准备

标准数据布局：

```text
datasets/<name>/
  images/{train,val,test}/*.jpg
  points/{train,val,test}/*.txt
```

每个点标注文件每行包含归一化的 `x y`。数据准备脚本：

```bash
python -m scripts.data.prepare_all
# 或按数据集运行 scripts/data/ 下对应 prepare_*.py
```

`PointDataset` 训练时使用缩放、裁剪、翻转、旋转和颜色增强；验证时使用保持纵横比的 letterbox。

## 训练

单数据集：

```bash
python -m scripts.training.train_moe \
    --weights yolo11m.pt \
    --data-root datasets/shanghaitech_AB \
    --native-references 1,4,16 \
    --native-warmup-epochs 5 \
    --save-dir runs/native_multiscale
```

联合训练：

```bash
python -m scripts.training.train_all \
    --weights yolo11m.pt \
    --save-dir runs/native_multiscale_all \
    --epochs 100
```

competition 的候选预筛选按 Expert 平衡：`match_top_k` 名额由 E0/E1/E2 平均分配，再执行全局 Hungarian。matching cost 默认是：

```text
5.0 × normalized_position_error - 0.25 × confidence
```

可用 `--match-position-weight` 和 `--match-confidence-weight` 调整，权重会写入 checkpoint；competition 阶段关闭 full-GT expert-only MAE，避免把 Expert 负责的局部 GT 数量误读为完整图像计数能力。

联合训练可重复指定：

```text
--dataset NAME=ROOT:TRAIN[:TRAIN2...]:EVAL
```

训练阶段：

1. 前 `--native-warmup-epochs` 个 epoch：E0/E1/E2 各自独立 Hungarian matching；同一个 GT 可分别成为三个 Expert 的正样本。
2. 后续 epoch：三个 Expert 的候选合并为一个 pool，对每个 GT 做一次全局 Hungarian matching；只有 winner candidate 为正样本。
3. 总损失：`L_cls + 5 * L_point + 0.05 * L_count`。

每个 epoch 记录：

- E0/E1/E2 GT winner 比例
- matched mean distance
- matched confidence
- positive count
- 每个数据集的上述统计
- warmup 阶段的 E0/E1/E2 expert-only MAE

checkpoint：

```text
best_native.pt
last.pt
```

`best_native.pt` 只在 global competition 阶段按 weighted normalized MAE 选择；warmup 阶段只保存 `last.pt`。

## 评估

单数据集 ShanghaiTech Part A/B：

```bash
python test_each_dataset.py \
    --data-root datasets/shanghaitech_AB \
    --checkpoint runs/native_multiscale/best_native.pt \
    --split val \
    --out-dir runs/native_multiscale/test_eval
```

跨数据集：

```bash
python -m scripts.evaluation.evaluate_datasets \
    --checkpoint runs/native_multiscale_all/best_native.pt \
    --batch-size 8 \
    --dataset shanghaitech=datasets/shanghaitech_AB:val \
    --dataset jhu=datasets/jhu_crowd:val \
    --dataset qnrf=datasets/ucf_qnrf:test \
    --dataset cc50=datasets/ucf_cc50:fold0_test \
    --out-dir runs/native_multiscale_all/eval
```

计数口径始终是所有 native candidates 的 `sum(sigmoid(logits))`。评估不使用置信度阈值作为 MAE 计数口径，也不使用 NMS。

## 可视化

单图：

```bash
python -m scripts.visualization.predict_moe \
    --image path/to/image.jpg \
    --checkpoint runs/native_multiscale/best_native.pt
```

批量：

```bash
python -m scripts.visualization.predict_moe_batch \
    --data-root datasets/shanghaitech_AB \
    --split val \
    --checkpoint runs/native_multiscale/best_native.pt \
    --out-dir runs/native_multiscale/val_pred
```

颜色固定为：红色 E0/P3、绿色 E1/P4、蓝色 E2/P5。颜色来自模型输出的真实 `expert_indices`。
