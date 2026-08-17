# CrowdCounting

基于 YOLO11 的人群计数项目。模型不回归边界框，只预测人头中心点，支持多数据集。

两个分支：

1. **点级 Scale-MoE Head 分支（当前推荐）**：保留 YOLO11 Backbone+Neck、删除原始
   Detect Head，替换为统一候选点网格（P3 网格 × K 参考点）上的三专家 MoE 点检测头。
   由 `scripts/training/train_moe.py` 训练，支持 JHU-Crowd++ 等多数据集。
2. **v4 PointDetect 分支（旧实现）**：在 YOLO11 各尺度（P2–P5）上添加逐尺度点检测头，
   走 ultralytics 训练管线。保留用于对照实验，不再推荐新训练。

## 环境

```bash
pip install -r requirements.txt
# GPU 机器装 CUDA 版 torch（实测环境 Python 3.13 / torch 2.12.1+cu130 / ultralytics 8.4.71）:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

运行约定：**始终从项目根目录以模块方式执行**（`python -m scripts.xxx`），避免相对导入
与工作目录问题。除标注 argparse 的脚本外，其余脚本的路径/参数写在 `__main__` 硬编码配置里，
切换数据集或权重时直接改对应脚本底部即可。

## 目录

- `models/`：模型结构、锚点生成和损失函数。
  - `yolo11_pyramid.py`：`YOLO11Pyramid`，保留 Backbone+Neck、移除 Detect Head，
    输出 P3/P4/P5（层索引与步长从 Detect Head 动态读取，不硬编码）。
  - `moe_point_head.py`：`MoEPointHead`，三个专家（局部/中层/大范围感受野）+ 点级 Router。
  - `yolo11_moe_point.py`：`YOLO11MoEPoint`，Backbone+Neck + MoE Point Head 组合网络。
  - `point_moe_loss.py`：`PointMoELoss`，匈牙利匹配 + Focal 分类 + Smooth L1 定位 + 计数 + 尺度路由监督。
  - `crowd_model.py` / `modules.py` / `loss.py` / `yolo11-crowd.yaml`：v4 分支（`CrowdCountingModel`
    + `PointDetect` + `CrowdPointLoss`）。
- `scripts/data/`：数据集下载（6 个公开数据集）、ShanghaiTech 转换、离线增强与点标注转换。
- `scripts/training/`：`train_moe.py`（Scale-MoE，argparse）与 v4/标准模型训练脚本。
- `scripts/evaluation/`：人数（MAE/RMSE）、点定位（P/R/F1）与模型对比评估。
- `scripts/visualization/`：单图/批量预测、GT-Pred 对比与样本绘制。
- `scripts/diagnostics/`：不修改数据的匹配逻辑检查工具。
- `test_each_dataset.py`（根目录）：Scale-MoE 模型按 Part A/B 分组的评估脚本。
- `runs/`：训练、评估与可视化输出（运行生成）。

## 数据集下载

`scripts/data/` 下提供各公开人群计数数据集的下载脚本（仅用标准库，带进度条，
从项目根目录以模块方式执行）。原始数据统一落在 `data/`。

| 数据集                      | 脚本                                                             | 图像数量                                    | 标注/说明                                                 |
| --------------------------- | ---------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| JHU-Crowd++ (Sindagi 2019)  | `python -m scripts.data.download_jhu_crowd`                    | 4,372（train 2,272 / val 500 / test 1,600） | 约 151 万点标注；单图最多 25,791 人；含恶劣天气与无人图像 |
| ShanghaiTech A (Zhang 2016) | `python -m scripts.data.download_shanghaitech`（A+B 一并下载） | 482（官方 train 300 / test 182）            | 官方 train 再切 270 train / 30 val                        |
| ShanghaiTech B (Zhang 2016) | 同上                                                             | 716（官方 train 400 / test 316）            | 官方 train 再切 360 train / 40 val                        |
| UCF-CC-50 (Idrees 2013)     | `python -m scripts.data.download_ucf_cc50`                     | 50                                          | 5 折 outer test；每折 36 train / 4 val / 10 test          |
| UCF-QNRF (Idrees 2018)      | `python -m scripts.data.download_ucf_qnrf`                     | 1,535（官方 Train 1,201 / Test 334）        | 官方 Train 再切约 1,081 train / 120 val；约 125 万点标注  |

各脚本均支持 `--data-dir`（默认 `data/`）与 `--keep-zip`/`--keep-rar`；已存在目标目录时
自动跳过；解压后打印实际顶层结构。大文件支持**断点续传**：下载中断会保留 `.part` 文件，
重跑脚本自动从断点继续（并按 Content-Length 校验完整性）。来源说明（脚本内亦有注释）：

- JHU-Crowd++：官方需在 crowd-counting.com 填表，脚本直接走社区公开 Google Drive
  镜像（jhu_crowd_v2.0.zip，约 2.87 GB），自动处理 Drive 大文件确认页。
- ShanghaiTech：官方 Dropbox 镜像（约 166 MB），可 `--url` 换源。
- UCF-CC-50 / UCF-QNRF：官方 CRCV 直链（rar / zip）。

## 快速开始（Scale-MoE 分支）

### 1. 准备数据

以 ShanghaiTech 为例（其余数据集需自行把原始数据放到 `data/` 下并准备点标注）：

```bash
# 0. 下载官方数据集（约 166 MB）
python -m scripts.data.download_shanghaitech

# 1. 合并 A/B 生成 datasets/shanghaitech_AB：
#    images/{train,val,test}/*.jpg + labels/{train,val,test}/*.txt
#    每个 Part 的官方 train 独立切 90/10；官方 test_data 保留为 test
python -m scripts.data.prepare_combined

# 2. 虚拟框标签转纯点标签（labels/ 每行第 2、3 列 -> points/ 每行 "nx ny"）
python -m scripts.data.prepare_point_labels
```

`PointDataset`（`scripts/data/point_dataset.py`）读取 `images/ + points/`：训练走 5 步在线增强
（随机缩放 0.8–1.2、随机裁剪 `crop_size=640`、翻转 p=0.5、亮度 α∈[0.8,1.2]、β∈[−20,20]）；
验证走 letterbox（保持纵横比 + 居中填充 114，不改变人的尺度）。

其余数据集同样转换为标准布局（纯标准库，无需 scipy/h5py/cv2，`_matlab_utils.py`
内置 Matlab v5 .mat 解析与 JPEG/PNG 尺寸读取）：

```bash
# JHU-Crowd++（gt 为逐图 txt，取 x y 两列）: datasets/jhu_crowd/{train,val,test}
python -m scripts.data.prepare_jhu
# UCF-QNRF（2024 版扁平布局 Train/Test，annPoints 点标注）: datasets/ucf_qnrf/{train,val,test}
python -m scripts.data.prepare_qnrf
# UCF-CC-50（5 折，seed=0；每折 36 train / 4 val / 10 test）:
#   datasets/ucf_cc50/{fold0_train,fold0_val,fold0_test,...}
python -m scripts.data.prepare_ucf_cc50
```

每个数据集目录都含 `images/ + labels/ + points/ + dataset.yaml`，images 与 points 一一对应。

**一键流程**（转换全部 + 联合训练 + 跨数据集评估）：

```bash
# 一个命令转换全部数据集（已存在则跳过，--force 强制重转）
python -m scripts.data.prepare_all

# 在所有数据集上联合训练（按图片自然采样 + 逐数据集验证）
python -m scripts.training.train_all \
    --weights yolo11n.pt \
    --crop-size 640 \
    --batch-size 32 \
    --save-dir runs/moe_point_all

# 跨数据集分组评估（默认从 checkpoint 读取 crop_size）
python -m scripts.evaluation.evaluate_datasets \
    --checkpoint runs/moe_point_all/best_hard.pt \
    --batch-size 8 \
    --dataset shanghaitech=datasets/shanghaitech_AB:val \
    --dataset jhu=datasets/jhu_crowd:val \
    --dataset qnrf=datasets/ucf_qnrf:test \
    --dataset cc50=datasets/ucf_cc50:fold0_test
```

### 2. 训练

```bash
python -m scripts.training.train_moe \
    --weights yolo11n.pt \
    --data-root datasets/shanghaitech_AB \
    --save-dir runs/moe_point
```

常用参数（默认值见 `python -m scripts.training.train_moe --help`）：

| 参数                              | 默认        | 说明                                                          |
| --------------------------------- | ----------- | ------------------------------------------------------------- |
| `--crop-size`                   | 640         | 训练/验证裁剪尺寸                                             |
| `--batch-size`                  | 8           |                                                               |
| `--epochs`                      | 100         |                                                               |
| `--backbone-lr` / `--head-lr` | 1e-4 / 1e-3 | YOLO 主干 / MoE Head 两组 AdamW 学习率                        |
| `--num-references`              | 4           | 每网格参考点数 K（1/4/9）                                     |
| `--freeze-epochs`               | 3           | 前 N 个 epoch 冻结 YOLO 主干                                  |
| `--route-weight`                | 0.15        | 尺度路由监督 macro CE 权重                                    |
| `--match-top-k`                 | 2000        | 匈牙利匹配候选点上限（K=max(K, n_gt)）                        |
| `--force-hard-epoch`            | None        | 强制切换硬路由的 epoch；默认由 Router 毕业条件决定            |
| `--resume`                      | None        | 从 checkpoint 恢复（正式修复实验应从`yolo11n.pt` 新开 run） |

训练要点：

- **软/硬路由两阶段**：默认不按 epoch 强制切硬路由，而是看 Router 毕业条件——验证集
  路由混淆矩阵 E0/E1/E2 行 recall 连续 3 轮 ≥ 0.60/0.40/0.30 且 macro recall ≥ 0.50，
  下一轮起切 Top-1 硬路由（Straight-Through）。可用 `--force-hard-epoch` 覆盖。
- **温度调度**：软阶段 2.0 →（epoch 15 处 1.3）→ 1.0；硬阶段 1.0 → 0.5（20 个 epoch）。
- **Router 梯度隔离与防饿死**：epoch < `--router-grad-epoch`(15) 时，cls/point/count 不经过
  gate 向 Router 回传；`--expert-uniform-floor` 默认 0.3 让每个专家获得最低 task gradient。
- **日志**：每 epoch 输出 `route`（batch-level macro CE）、`T`、`hard_route`、`router_grad`、
  weighted/macro normalized MAE、GT target/gate/val hard usage、混淆矩阵、各类 support、
  recall/macro recall 与 Router entropy。
- **checkpoint**：`best_soft.pt`、`best_hard.pt` 分别记录两个 phase 的最优模型，`last.pt`
  保存完整训练状态。主选模指标是按验证图片数加权的 normalized MAE；macro normalized MAE
  仅作跨数据集泛化参考。checkpoint 同时记录 crop size、参考点数、温度 schedule、hard 状态、
  Router 梯度 epoch 与 scale centers。Router 未毕业时不会生成 `best_hard.pt`，训练结束会告警。
  未传 `--save-dir` 时输出到 `runs/moe_point_<时间戳>/` + `train.log`
  （每次启动自动带时间戳，互不覆盖；`--resume` 时沿用原 run 目录）。

### 3. 评估与推理

```bash
# 单图推理（红/绿/蓝 = E3 精细 / E4 中层 / E5 大范围）
python -m scripts.visualization.predict_moe \
    --image path/to/img.jpg \
    --checkpoint runs/moe_point/best_hard.pt

# 验证集批量推理：images/*_pred.jpg + predictions.csv（逐图计数）+ summary.json（MAE/RMSE）
python -m scripts.visualization.predict_moe_batch \
    --data-root datasets/shanghaitech_AB \
    --checkpoint runs/moe_point/best_hard.pt \
    --out-dir runs/moe_point/val_pred

# 分 Part 评估（Part A/B/overall 的 MAE/RMSE；hard 路由 + Σsigmoid 计数口径）
python test_each_dataset.py \
    --data-root datasets/shanghaitech_AB \
    --checkpoint runs/moe_point/best_hard.pt
```

MoE 计数口径：全部候选点 `logits.sigmoid()` 求和（无需置信度阈值/去重，与评估一致）；
soft 验证使用当前训练温度，hard 使用 0.5（argmax 不受温度影响）。测试脚本默认读取
checkpoint 的 `crop_size`，若 checkpoint 不是 hard phase 会显式告警。

### 3.5 跨数据集分组评估

```bash
python -m scripts.evaluation.evaluate_datasets \
    --checkpoint runs/moe_point_all/best_hard.pt \
    --batch-size 8 \
    --dataset shanghaitech=datasets/shanghaitech_AB:val \
    --dataset jhu=datasets/jhu_crowd:val \
    --dataset qnrf=datasets/ucf_qnrf:test \
    --dataset cc50_fold0=datasets/ucf_cc50:fold0_test \
    --out-dir runs/eval_datasets
```

对每个 `NAME=ROOT:SPLIT` 分别评估并输出该数据集的 MAE/RMSE/GT/Pred：
`<out-dir>/<name>/predictions.csv` + `summary.json`，另生成合并的
`<out-dir>/summary.json` 与控制台汇总表。UCF-CC-50 5 折需逐折指定
（fold0_test..fold4_test），汇总均值自行取。

### 4. 判断专家坍缩

日志 `target` 是 GT 尺度目标的 argmax 分布（来自真实最近邻间距），`gate` 是预测路由分布，
`hard_route` 统计硬路由下置信度 >0.5 候选的专家使用率。密集场景大量使用 E3 是**正确的尺度
专家化**，不是坍缩。若某专家几乎从不被选中（其尺度区间内几乎没有 GT 样本），应检查
`scale_centers`（默认 10/20/40 px，`models/point_moe_loss.py` 中可调）是否与数据尺度匹配，
以及 Router 是否一直未毕业（见日志 `route` 与混淆矩阵）。

## v4 PointDetect 分支（旧）

ultralytics 管线，硬编码配置，运行后修改脚本底部即可：

```bash
# 训练（imgsz=1024、epochs=200、加载本地 yolo26n.pt 的 Backbone 0–10 层）
python scripts/training/train_custom_v4.py
# 对照：标准 YOLO11n 微调 baseline（imgsz=640）
python scripts/training/train_standard.py

# 评估：人数（MAE/RMSE，扫置信度阈值）/ 点定位（P/R/F1，容忍距离 15px）/ 逐图明细
python scripts/evaluation/count.py
python scripts/evaluation/localization.py
python scripts/evaluation/detailed.py
python scripts/evaluation/compare_localization.py

# 可视化
python scripts/visualization/predict.py
python scripts/visualization/compare_custom.py
```

注意：v4 用检测框数做计数、用框中心做点匹配，训练日志里的标准 YOLO `Box(P/R/mAP)` 因
虚拟框（1×1 px）与标签框（宽高 1%）IoU 近乎恒为 0，**不能**用于评价该模型；请使用点级
MAE/RMSE 与 P/R/F1，并固定置信度阈值、匹配容忍距离和 `max_det` 后再比较。

模型与损失的设计细节见：

- [`crowd_counting_model_summary.md`](./crowd_counting_model_summary.md)（当前 Scale-MoE + v4 附录）
- [`loss_design_explanation.md`](./loss_design_explanation.md)（`PointMoELoss` 详解 + v4 附录）
- [`docs/datasets.md`](./docs/datasets.md)（数据集下载/转换/训练接入/跨数据集评估指南）
