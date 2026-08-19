# 项目提交总结与研究状态

> 生成日期：2026-08-18。基于 git 历史（60 个提交，`42b15a9` … `0047056`）与最新训练日志。

## 1. 当前状态：研究已暂停

**本项目的研究工作目前已暂停。** 暂停时的最新进展是 2026-08-18 15:36 提交的
`0047056 Implement D2 Drop-1 Soft Top-2 MoE routing`（D2 方案，当前推荐路线）。

最后一次训练为联合训练任务（`runs/moe_point_all_20260818_201021`，4 个数据集联合、
batch 8、100 epochs，20:10 启动），日志仅记录到 Epoch 2——Router 尚未激活，
仍处于 warm-up 随机 Drop-1 阶段，无法据此判断新方案效果。暂停时该训练仍在后台运行。

**目前未解决的主要问题是专家坍缩（Expert Collapse）**，详见第 3 节。

## 2. 全部提交总结（按阶段）

### 阶段 0 — v2/v3/v4 PointDetect 分支（2026-08-08，2 个提交）

| 提交 | 说明 |
| --- | --- |
| `42b15a9` feat:initial commit | 初始代码：YOLO11 自定义点检测模型（`CrowdCountingModel` + `PointDetect` + `CrowdPointLoss`，不回归框、只预测人头中心点）、v2/v3/v4 训练脚本、人数/定位/对比评估、可视化与训练产物；标准 YOLO11n baseline |
| `902a774` refactor:Struggle Change | 目录重构：扁平脚本整理进 `scripts/{data,training,evaluation,visualization,diagnostics}` 包；`loss.py` 修改 |

### 阶段 1 — 转向点级 Scale-MoE Head（2026-08-13，7 个提交）

| 提交 | 说明 |
| --- | --- |
| `ae229b9` feat(models): 点级 Scale-MoE Head 模型 | 主分支建立：`YOLO11Pyramid`（Backbone+Neck，动态读取 Detect 层索引）、`MoEPointHead`（三专家：局部/中层/大范围感受野 + 点级 Router）、`PointMoELoss` |
| `af6c67a` feat(data): 点监督数据集与点标签转换脚本 | 点标签转换管线 |
| `d58ba09` feat(training): MoE 点检测训练脚本 | 温度调度 / 硬路由切换 / 冻结主干 |
| `a8b87b0` feat(visualization): MoE 点检测推理与三专家彩色可视化 | 单图推理 + 三专家着色 |
| `aca6ea2` feat(data): ShanghaiTech 数据集下载脚本 | 官方 Dropbox 镜像 |
| `9bb31a9` docs: README 补充 MoE 分支说明 | 数据准备/训练/推理流程 |
| `3ac2f5e` chore: 忽略 .idea 与 __pycache__ | |

### 阶段 2 — MoE 训练打磨 + 多数据集扩展（2026-08-15，30 个提交）

| 提交 | 说明 |
| --- | --- |
| `4816409` feat:add requirements.txt | 依赖清单 |
| `a9fc3c9` fix:expand bs | 扩大 batch size |
| `ff23d82` chore: add tqdm dependency | 训练进度条 |
| `25b3378` fix(moe): address design review of Scale-MoE training | 按设计评审修正训练 |
| `f95b478` feat(visualization): batch validation inference | letterbox 预处理批量推理 |
| `2febfa1` fix(training): always save best.pt | 按阶段指标保存 best |
| `92bb6cd` fix(loss): restore point localization gradient scale | 定位梯度量级修复（Smooth L1 β=0.02） |
| `e4baceb` fix(loss): prune Hungarian matching candidates to top-K | 匈牙利匹配候选剪枝（防密集图卡死） |
| `3e22ddd` feat(visualization): soft count mode | 与训练口径一致的软计数 |
| `afcbed3` docs: crop-size 640 示例 | |
| `4e16262` feat(visualization): 加载时打印 checkpoint epoch/best_mae | |
| `e6e9f58` fix(visualization): remove stale pred_count overwrite | |
| `285c943` feat(training): per-epoch gate/target/usage diagnostics | 逐 epoch 路由诊断 |
| `f3073e8` feat(visualization): probability-field heatmap overlay | 概率场热图叠加 |
| `cfede2f` fix(training): resume no longer wipes historical best | 断点恢复保留历史 best |
| `5b8e7bd` refactor(logging): print → logging 文件输出 | |
| `07dcef5` fix(training): foreground-only hard-usage stats | 只统计前景硬路由使用率 + 全新运行备份 |
| `0778494` fix(training): expandable_segments | 避免解冻主干 OOM 崩溃 |
| `e3ed849` fix(training): checkpoint ordering, epoch-averaged losses, routing confusion matrix | |
| `0c96cbf` fix(moe): router gradient isolation, macro-balanced route loss, graduation-based hard routing | **路由梯度隔离 + 宏观平衡路由损失 + 毕业式硬路由** |
| `cb88527` feat(data): dataset download scripts | 断点续传 + 完整性校验 |
| `e64e110` feat(data): stdlib mat v5 parser | JHU/QNRF/CC-50 纯标准库转换 |
| `7c23e6a` feat(eval): cross-dataset grouped evaluation runner | 跨数据集分组评估 |
| `5bf6e05` docs: dataset guide, README refresh, model/loss docs rewrite | |
| `ea21ad1` chore: ignore macOS .DS_Store | |
| `4c5197c` feat(data): one-command all-dataset conversion (`prepare_all`) | |
| `adab734` feat(training): joint all-dataset training (`train_all`) | 复用 train_moe 超参 |
| `46e0d27` docs: one-command convert and joint-train pipeline | |
| `88b762c` feat(training): timestamped save-dir | 每次训练时间戳目录 |

### 阶段 3 — 评估口径对齐与数据清理（2026-08-16 ~ 08-17，16 个提交）

| 提交 | 说明 |
| --- | --- |
| `c271ed8` fix: isolate validation and test splits | 隔离验证集与测试集 |
| `8e48294` feat:删除无效运行文件 | 清理无效产物 |
| `67f5741` docs:脚本命令增强 | |
| `71e65b0` fix: align evaluation with checkpoint configuration | 评估对齐 checkpoint 配置 |
| `ccdc776` fix: balance Router loss and protect experts | **平衡 Router 损失、保护专家**（对抗坍缩） |
| `e01f526` feat: add weighted selection and Router diagnostics | 加权选择 + Router 诊断 |
| `ca54364` config: set default input size to 640 | |
| `3c8c009` fix: align Router diagnostics and small-image scaling | |
| `2d7d84e` fix: restore target diagnostics initialization | |
| `b056513` emove data which larger than 1000. | 删除人数 >1000 的样本提升训练效率（提交信息拼写笔误） |
| `98bec49` feat(data): 增强支持旋转，离线增强同步输出 points/ 标签 | |
| `76721dc` feat(train): CUDA 设备架构预检 | fail-fast 拒绝不匹配的 torch wheel |
| `f06ea1f` feat(predict): 从 checkpoint args 恢复模型结构参数 | |
| `7c07538` config: 默认权重 yolo11n→yolo11m，hidden_channels 128→256 | |
| `10b0ec5` docs: 同步数据增强与默认配置说明 | |

### 阶段 4 — 路由机制演进：对抗专家坍缩（2026-08-18，7 个提交，最新）

| 提交 | 说明 |
| --- | --- |
| `e0acb96` Implement task-only MoE router training | **弃用全部 Router loss**（route/balance/sharp），Router 只收 task-only 梯度 |
| `01adddb` Implement Hard-Only Gumbel-ST MoE routing | **H0 方案**：Gumbel-ST 硬路由 |
| `0f7078c` Fix H0 matched routing diagnostics | H0 匹配路由诊断修复 |
| `b7a1f4f` Clarify H0 routing diagnostics | H0 诊断口径澄清 |
| `98bd890` feat: 为 train_moe 和 train_all 添加 TensorBoard 支持 | |
| `1c30d78` chore:smaller batch size | 减小 batch size（README 默认值） |
| `0047056` Implement D2 Drop-1 Soft Top-2 MoE routing | **D2 方案（当前推荐）**：candidate 级随机 Drop-1 + masked softmax 软 Top-2 |

## 3. 主要问题：专家坍缩（Expert Collapse）

### 3.1 现象

MoE 点检测头的三专家（E0 局部 / E1 中层 / E2 大范围感受野）由 Router 分配权重。
**专家坍缩**指路由分布失去多样性：Router 把绝大多数（乃至全部）候选点路由到
同一个专家，其余专家收不到有效梯度、逐渐变成"死专家"（starvation / dead expert）；
或三个专家共享 P3 细节后退化成近似副本。

观测指标（`train_moe.py` / `train_all.py` 逐 epoch 日志）：

- `full3 matched probability mean`：应接近 33/33/33；坍缩时偏向一个专家。
- `full3 entropy / margin`：均匀时 entropy=ln3≈1.0986、margin=0；坍缩时 entropy 下降。
- `full3 deterministic Top-1 usage`：坍缩时单专家接近 100%。
- `train sampled usage / matched train gate Top-1`：训练阶段实际使用分布。
- **Expert-only MAE**（默认每 5 epoch）：单独跑 E0/E1/E2-only 对比，判断 starvation
  是否解决——这是区分"Router 坍缩"与"专家本身退化"的关键。

最新一次训练（`runs/moe_point_all_20260818_201021`，Epoch 1–2，Router 尚未激活）：
概率均匀 33/33/33、entropy=1.0986、margin=0，处于 warm-up 随机 Drop-1 阶段；
但 `train sampled usage = E0:66.7% E1:33.3% E2:0.0%`——E2 在采样使用率中已为 0，
说明坍缩风险在 Router 激活前就已埋下（随机 Drop-1 的 argmax 打平取样 + 初始化偏置）。

### 3.2 尝试过的对策（演进路径）

| 时间 | 方案 | 提交 | 结果/去向 |
| --- | --- | --- | --- |
| 08-15 | Router 梯度隔离 + 宏平衡路由损失 + 毕业式硬路由 | `0c96cbf` | 早期方案，后被简化 |
| 08-17 | 平衡 Router 损失 + 保护专家 | `ccdc776` | 加路由损失约束，仍不稳 |
| 08-18 | **Task-only Router**：删光 route/balance/sharp 损失，Router 只收 task-only 梯度 | `e0acb96` | 去掉人为路由损失，避免与任务损失打架 |
| 08-18 | **H0**：Hard-Only Gumbel-ST 硬路由 | `01adddb`（+`0f7078c`/`b7a1f4f` 诊断修复） | 硬路由方案，诊断口径修正后未定论 |
| 08-18 | **D2（当前）**：candidate 级随机 Drop-1 + masked softmax 软 Top-2；训练恒 2 个活跃专家，双概率记录（完整 softmax 用于诊断、gates 用于实际混合） | `0047056` | **当前推荐**，暂停时训练刚起步，尚无结论 |

### 3.3 当前结论

- 专家坍缩**尚未解决**；D2 是暂停前最后、也是当前推荐的尝试方向，其设计（随机
  Drop-1 保证每个专家在训练中持续暴露、soft Top-2 保留确定性推理、task-only
  梯度避免人为路由损失干扰）正是针对坍缩的针对性改造。
- 暂停时 D2 训练仅跑到 Epoch 2（Router 冻结期），不足以评估；后续恢复研究时，
  需观察 Router 激活（Epoch 7+）后的 Top-1 usage 与 Expert-only MAE。
- 附注：`README.md` 2026-08-17 备注已按训练效率删除人数 >1000 的样本
  （`b056513`），后续所有实验的数据口径都基于过滤后的数据集。

## 4. 仓库结构速览

```
models/            YOLO11Pyramid + MoEPointHead + PointMoELoss（主分支）；
                   crowd_model.py + loss.py（v4 旧分支，仅对照）
scripts/data/      下载 / 转换 / 增强 / prepare_all
scripts/training/  train_moe.py（D2 主训练）/ train_all.py（联合训练）
scripts/evaluation/ 跨数据集评估 evaluate_datasets.py
scripts/visualization/ predict_moe / predict_moe_batch
scripts/diagnostics/  匹配逻辑检查
test_each_dataset.py  按 Part 分组评估
runs/              训练输出（时间戳目录 + TensorBoard）
```

## 5. 恢复研究的切入点

1. 跑完/重跑 D2 训练至 Router 激活期，读 `Top-1 usage`、`entropy`、`expert-only MAE`
   三项坍缩指标（TensorBoard 已接入，`runs/*/tensorboard`）。
2. 若坍缩仍发生：优先排查 Router 初始化偏置（当前全 0 → 均匀起步）与 warm-up
   期随机 Drop-1 的采样偏置（E2 使用率为 0 的迹象）。
3. 对照实验基线：`full3_soft`（完整三专家）与 `top1`（确定性单专家）MAE 已在
   每 epoch 记录，可直接判断 Router 是否带来增益。
