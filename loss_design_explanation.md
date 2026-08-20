# Native Multiscale Point Loss

## 1. 候选点

YOLO11 输出 P3/P4/P5。三个 Expert 在各自原生特征层生成候选：

```text
P3 -> E0, K=1
P4 -> E1, K=4
P5 -> E2, K=16
```

三层在 640×640 输入下各自产生 6400 个候选，reference spacing 都约为 8px。三个候选子池随后 concat 成一个 native candidate pool。

## 2. 两阶段 matching

### 2.1 Independent warmup

前 `native_warmup_epochs` 个 epoch 对每个 Expert 分别执行 Hungarian matching。每个 Expert 都需要独立学习：

```text
classification + localization + count
```

因此同一个 GT 在 warmup 阶段可以分别给 E0、E1、E2 一个正样本，避免随机初始化时某个 Expert 先占优势而饿死其它 Expert。

### 2.2 Global competition

warmup 之后，三个候选子池合并后只做一次全局 Hungarian matching：

```text
cost = 5 × normalized_L1_position_error - sigmoid(confidence_logit)
```

每个 GT 只有一个 winner candidate。winner 的 Expert 统计为 `GT winner`，其它 Expert 对重复位置的候选保持负样本。

损失中的匹配索引由 detached cost 得到，但定位损失仍作用于原始预测张量，保持反向传播。

## 3. Task loss

```text
L = L_cls + 5 × L_point + 0.05 × L_count
```

- `L_cls`：sigmoid focal loss，未匹配候选为负样本。
- `L_point`：matched point 的 normalized Smooth L1，`beta=0.02`。
- `L_count`：

  ```text
  abs(sum(sigmoid(logits)) - n_gt) / (n_gt + 1)
  ```

计数使用整个 concat candidate pool，不使用 NMS 或 confidence threshold。

## 4. 诊断

每个 epoch 保存和记录：

```text
E0/E1/E2 GT winner percentage
E0/E1/E2 matched mean distance
E0/E1/E2 matched confidence
E0/E1/E2 positive count
```

联合训练额外按 ShanghaiTech、JHU、QNRF、CC50 分别记录这些统计。E0/E1/E2-only MAE 仅用于观察各 Expert 的独立能力，不参与 matching 监督。
