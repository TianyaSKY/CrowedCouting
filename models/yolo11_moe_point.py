from __future__ import annotations

import torch
import torch.nn as nn

from models.moe_point_head import MoEPointHead
from models.yolo11_pyramid import YOLO11Pyramid


class YOLO11MoEPoint(nn.Module):
    """YOLO11 Backbone+Neck with the native multiscale point head."""

    architecture = "native_multiscale"

    def __init__(
        self,
        weights: str = "yolo11m.pt",
        hidden_channels: int = 256,
        native_references: tuple[int, int, int] = (1, 4, 16),
    ) -> None:
        super().__init__()
        self.yolo = YOLO11Pyramid(weights)
        self.point_head = MoEPointHead(
            feature_channels=self.yolo.output_channels,
            output_strides=self.yolo.output_strides,
            hidden_channels=hidden_channels,
            num_references=native_references,
            offset_range=2.0,
        )

    def forward(
        self,
        image: torch.Tensor,
        routing_mode: str = "native",
        expert_index: int | None = None,
    ) -> dict[str, object]:
        features = self.yolo(image)
        return self.point_head(
            features,
            routing_mode=routing_mode,
            expert_index=expert_index,
        )
