from __future__ import annotations

import torch
import torch.nn as nn

from models.yolo11_pyramid import YOLO11Pyramid
from models.moe_point_head import MoEPointHead


class YOLO11MoEPoint(nn.Module):
    """YOLO11 Backbone+Neck + 点级 Scale-MoE Head 的组合网络。

    输入图像直接输出统一的候选点集合（P3 网格 x K 参考点），
    不再保留任何原始 Detect 头或额外的逐尺度检测头。
    """

    def __init__(
        self,
        weights: str = "yolo11m.pt",
        hidden_channels: int = 256,
        num_references: int = 4,
    ) -> None:
        super().__init__()

        self.yolo = YOLO11Pyramid(weights)

        self.point_head = MoEPointHead(
            feature_channels=self.yolo.output_channels,
            hidden_channels=hidden_channels,
            num_references=num_references,
            output_stride=self.yolo.output_strides[0],
            offset_range=2.0,
        )

    def forward(
        self,
        image: torch.Tensor,
        temperature: float = 1.0,
        routing_mode: str = "full3_soft",
        router_grad: bool = True,
        expert_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.yolo(image)

        return self.point_head(
            features,
            temperature=temperature,
            routing_mode=routing_mode,
            router_grad=router_grad,
            expert_index=expert_index,
        )
