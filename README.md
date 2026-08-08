# CrowdCounting

基于自定义 PointDetect 头的人群计数项目。

## 目录

- `models/`：模型结构、锚点生成和损失函数。
- `scripts/data/`：ShanghaiTech 数据集转换与离线增强。
- `scripts/training/`：自定义模型、标准 YOLO 基线的训练与恢复训练。
- `scripts/evaluation/`：人数、定位和模型对比评估。
- `scripts/visualization/`：单图预测、结果绘制与可视化对比。
- `scripts/diagnostics/`：不修改数据的匹配逻辑检查工具。
- `runs/`：训练、评估与可视化输出（运行生成）。

## 运行方式

始终从项目根目录以模块方式执行，避免相对导入和工作目录问题：

```bash
python -m scripts.data.prepare_combined
python -m scripts.data.augment
python -m scripts.training.train_custom_v4
python -m scripts.evaluation.count
python -m scripts.evaluation.localization
python -m scripts.visualization.predict
```

每个脚本底部保留了原有的默认路径和参数；需要切换数据集、权重或阈值时，修改对应脚本的 `__main__` 配置即可。
