# 数据集指南

本项目支持 4 个公开人群计数数据集（ShanghaiTech A/B、JHU-Crowd++、UCF-QNRF、
UCF-CC-50）。原始数据在 `data/`，转换后的标准布局在 `datasets/`。
全部下载与转换脚本**只依赖标准库**（Matlab v5 .mat 解析与图片尺寸读取均为自实现，
无需 scipy/h5py/cv2）。

## 数据集一览

| 数据集 | 划分 | 点标注总数 | 说明 |
| --- | --- | --- | --- |
| ShanghaiTech A (Zhang 2016) | train 270 / val 30 / test 182 | A 241,677 | 官方 train 切 90/10，官方 test 留作最终评估 |
| ShanghaiTech B (Zhang 2016) | train 360 / val 40 / test 316 | B 88,488 | 官方 train 切 90/10，官方 test 留作最终评估 |
| JHU-Crowd++ (Sindagi 2019) | train 2,272 / val 500 / test 1,600 | 1,515,005 | 单图最多 25,791 人 |
| UCF-QNRF (Idrees 2018) | train 约 1,081 / val 约 120 / test 334 | 1,251,642 | 官方 Train 切 90/10，Test 留作最终评估 |
| UCF-CC-50 (Idrees 2013) | 5 折 × 36 train / 4 val / 10 test | 每折 63,974 | 极端密集；每折独立训练 |

标准布局（`datasets/<name>/`，images 与 points 一一对应）：

```text
datasets/<name>/
├── images/{split}/*.jpg
├── labels/{split}/*.txt     YOLO 虚拟框: 0 nx ny 0.010000 0.010000
├── points/{split}/*.txt     纯点: nx ny（归一化到 [0,1]）
└── dataset.yaml
```

## 1. 下载

```bash
python -m scripts.data.download_shanghaitech   # A+B 一并下载（约 166 MB）
python -m scripts.data.download_jhu_crowd      # Google Drive 镜像（约 2.87 GB）
python -m scripts.data.download_ucf_qnrf       # 官方 CRCV 直链（约 4.5 GB）
python -m scripts.data.download_ucf_cc50       # 官方 CRCV 直链（约 8 MB，rar）
```

- 大文件支持**断点续传**：中断保留 `.part`，重跑自动续传并按 Content-Length 校验。
- JHU-Crowd++ 官方需在 crowd-counting.com 填表；脚本走社区公开 Google Drive 镜像，
  自动处理大文件确认页。NWPU-Crowd 因官方 OneDrive 对本网络限流已移除。
- UCF-CC-50 为 rar 归档，自动用 bsdtar/unrar/unar/7z 解压。

## 2. 转换

```bash
# ShanghaiTech -> datasets/shanghaitech_AB（官方 train 切 train/val，官方 test_data 为 test）
python -m scripts.data.prepare_combined
python -m scripts.data.prepare_point_labels

# JHU-Crowd++ -> datasets/jhu_crowd（gt 为逐图 txt，取 x y 列作人头中心）
python -m scripts.data.prepare_jhu

# UCF-QNRF -> datasets/ucf_qnrf（2024 版扁平布局，annPoints 点标注）
python -m scripts.data.prepare_qnrf

# UCF-CC-50 -> datasets/ucf_cc50（5 折，seed=0 打乱，--seed 可覆盖）
python -m scripts.data.prepare_ucf_cc50
```

转换内部要点：

- **归一化**：`nx = clamp(x / w), ny = clamp(y / h)`，图片尺寸从 JPEG SOF / PNG IHDR
  直接读取，不缩放原图。
- **JHU 镜像怪癖**：个别 `.jpg` 实际为 PNG；gt 为 `x y w h flag flag` 的 txt，
  取前两列。这两点在 `_matlab_utils.py` 中已处理。
- **ShanghaiTech .mat** 顶层 `image_info` 为 cell 包裹的 struct，`location` 字段
  可能为 cell 或直接 Nx2；`mat_points()` 统一兼容。

## 3. 训练接入

**一键转换 + 联合训练**（推荐）：

```bash
python -m scripts.data.prepare_all        # 一个命令转换全部数据集（--force 重转）
python -m scripts.training.train_all \
    --weights yolo11m.pt \
    --crop-size 640 \
    --batch-size 8 \
    --save-dir runs/moe_point_all
```

`prepare_all` 幂等（已存在跳过）；`train_all` 把各数据集 train split 拼成
ConcatDataset（batch 内随机混合），逐数据集用内部 val 验证并输出各数据集 MAE，best 按
归一化 validation score 选取。默认覆盖 4 个数据集；`--dataset NAME=ROOT:TRAIN[:TRAIN2...]:EVAL`
可自定义（如 UCF-CC-50 逐折：`--dataset cc50=datasets/ucf_cc50:fold0_train:fold0_val`）。
训练期间默认拒绝任何 `test` split；如有特殊实验需显式传 `--allow-test-as-eval`。
超参数与 `train_moe` 共享 argparse，checkpoint 兼容 `evaluate_datasets` / `test_each_dataset`。

单数据集训练：

```bash
python -m scripts.training.train_moe \
    --weights yolo11m.pt \
    --data-root datasets/shanghaitech_AB \
    --save-dir runs/moe_point
```

| 数据集 | --data-root | 训练 split | 评估 split |
| --- | --- | --- | --- |
| ShanghaiTech A+B | datasets/shanghaitech_AB | train | val |
| JHU-Crowd++ | datasets/jhu_crowd | train | val |
| UCF-QNRF | datasets/ucf_qnrf | train | val |
| UCF-CC-50 | datasets/ucf_cc50 | fold{i}_train | fold{i}_val |

UCF-CC-50 为 5 折交叉验证：逐折训练（fold0..fold4），每折用对应 `fold{i}_train`
训练、`fold{i}_val` 选 best，训练结束后才用对应 `fold{i}_test` 评估，最终报告 5 折平均 MAE/RMSE。

## 4. 评估

```bash
# 跨数据集分组评估（hard 路由 + Σsigmoid 计数；默认读取 checkpoint crop_size）
python -m scripts.evaluation.evaluate_datasets \
    --checkpoint runs/moe_point_all/best_hard.pt \
    --batch-size 8 \
    --dataset shanghaitech=datasets/shanghaitech_AB:val \
    --dataset jhu=datasets/jhu_crowd:val \
    --dataset qnrf=datasets/ucf_qnrf:test \
    --dataset cc50_fold0=datasets/ucf_cc50:fold0_test \
    --out-dir runs/eval_datasets
```

输出：每个数据集 `predictions.csv` + `summary.json`，以及合并的 `summary.json`
（含全部数据集 MAE/RMSE 对照）。

```bash
# 单数据集分 Part 评估（ShanghaiTech A/B）
python test_each_dataset.py \
    --data-root datasets/shanghaitech_AB \
    --checkpoint runs/moe_point_all/best_hard.pt
```

## 附录：Matlab v5 .mat 解析（`scripts/data/_matlab_utils.py`）

所有数据集的 .mat 真值均为 Matlab v5 格式，`loadmat_v5()` 纯标准库实现，要点：

- **SDE 小数据元素**：前 4 字节高 16 位为 byte_count、低 16 位为类型（scipy 实测
  规则，与官方文档略有出入）；数据在后 4 字节内联。
- **零长度元素**只占 8 字节（无数据无 padding），跳过时不能按 ≤4 字节的 16 字节规则。
- **struct 字段**：字段名宽度（miINT32）后跟定宽字段名字符串，字段数 =
  总长度 ÷ 宽度；结构体数组按列优先、逐元素存字段。
- **PR 实部**按元素自身类型解包（MATLAB 可能用 uint16 存标量 count）。
- **miCOMPRESSED**：zlib 解压后为无文件头的元素流。
