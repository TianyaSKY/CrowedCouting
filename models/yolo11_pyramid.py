from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLO11Pyramid(nn.Module):
    """保留 YOLO11 Backbone + Neck，移除原始 Detect Head。

    输出与 Neck 最终融合一致的 P3、P4、P5 特征（对 yolo11n 为第 16/19/22 层，
    但这里直接从 Detect Head 的 ``f`` 字段动态读取，避免硬编码）。

    输出步长与候选点网格由 ``output_strides`` / ``output_channels`` 描述：
    - P3: stride 8（候选点所在层）
    - P4: stride 16
    - P5: stride 32
    """

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        super().__init__()

        yolo = YOLO(weights)
        base_model = yolo.model

        detect_head = base_model.model[-1]

        # 当前 YOLO11 中通常是 [16, 19, 22]，
        # 这里直接从 Detect Head 读取，避免硬编码。
        head_inputs = detect_head.f
        if isinstance(head_inputs, int):
            head_inputs = [head_inputs]

        self.output_indices = tuple(int(i) for i in head_inputs)

        # 移除最后的 Detect Head，仅保留 Backbone + Neck
        self.layers = nn.ModuleList(list(base_model.model[:-1]))

        # 保存原 YOLO 前向过程中需要缓存的层
        self.save_indices = set(getattr(base_model, "save", ()))
        self.save_indices.update(self.output_indices)

        stride = getattr(
            detect_head,
            "stride",
            torch.tensor([8.0, 16.0, 32.0]),
        )
        self.output_strides = tuple(int(v) for v in stride.tolist())

        # 动态获取不同 YOLO11 尺寸对应的通道数
        was_training = self.training
        self.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            features = self.forward(dummy)

        self.output_channels = tuple(
            feature.shape[1] for feature in features
        )

        self.train(was_training)

    def forward(
        self,
        image: torch.Tensor,
    ) -> list[torch.Tensor]:
        cached_outputs: list[torch.Tensor | None] = []
        pyramid_outputs: dict[int, torch.Tensor] = {}

        x = image

        for layer in self.layers:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = cached_outputs[layer.f]
                else:
                    x = [
                        x if index == -1 else cached_outputs[index]
                        for index in layer.f
                    ]

            x = layer(x)

            if layer.i in self.output_indices:
                pyramid_outputs[layer.i] = x

            cached_outputs.append(
                x if layer.i in self.save_indices else None
            )

        return [
            pyramid_outputs[index]
            for index in self.output_indices
        ]
