from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)

    while channels % groups != 0 and groups > 1:
        groups //= 2

    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(out_channels),
            nn.SiLU(inplace=True),
        )


class PointExpert(nn.Module):
    """单个点检测专家，输出 confidence、dx、dy。"""

    def __init__(
        self,
        channels: int,
        num_references: int,
        prior_probability: float = 0.01,
    ) -> None:
        super().__init__()

        self.num_references = num_references

        self.body = ConvBlock(channels, channels)

        # 每个参考点输出 confidence、dx、dy
        self.prediction = nn.Conv2d(
            channels,
            num_references * 3,
            kernel_size=1,
        )

        prior_bias = math.log(
            prior_probability / (1.0 - prior_probability)
        )

        nn.init.normal_(self.prediction.weight, std=0.01)

        with torch.no_grad():
            bias = self.prediction.bias.view(
                num_references,
                3,
            )
            bias[:, 0].fill_(prior_bias)
            bias[:, 1:].zero_()

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.body(feature)

        batch_size, _, height, width = feature.shape

        output = self.prediction(feature).view(
            batch_size,
            self.num_references,
            3,
            height,
            width,
        )

        confidence_logits = output[:, :, 0]
        offsets = output[:, :, 1:3]

        return confidence_logits, offsets


 
 
class MoEPointHead(nn.Module):
    """Native P3/P4/P5 point experts without a Router.

    Each expert predicts on its own feature map and owns a square reference
    grid.  The default reference counts are ``1/4/16`` for P3/P4/P5, so all
    three levels expose approximately the same eight-pixel reference spacing.
    The returned candidate pool is concatenated in expert order and carries
    ``expert_indices`` so matching and visualization can identify the real
    source expert.
    """

    architecture = "native_multiscale"

    def __init__(
        self,
        feature_channels: tuple[int, int, int],
        output_strides: tuple[int, int, int],
        hidden_channels: int = 256,
        num_references: tuple[int, int, int] = (1, 4, 16),
        offset_range: float = 2.0,
    ) -> None:
        super().__init__()

        if len(feature_channels) != 3 or len(output_strides) != 3:
            raise ValueError(
                "native_multiscale 需要三个 feature channel 和 stride"
            )
        if len(num_references) != 3:
            raise ValueError(
                "native_multiscale 需要三个 num_references"
            )

        self.num_experts = 3
        self.references_per_expert = tuple(
            int(value) for value in num_references
        )
        self.output_strides = tuple(
            int(value) for value in output_strides
        )
        self.effective_strides = tuple(
            stride / math.sqrt(references)
            for stride, references in zip(
                self.output_strides,
                self.references_per_expert,
            )
        )
        self.offset_range = float(offset_range)

        c3, c4, c5 = feature_channels
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, hidden_channels, 1, bias=False),
            make_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(c4, hidden_channels, 1, bias=False),
            make_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.proj5 = nn.Sequential(
            nn.Conv2d(c5, hidden_channels, 1, bias=False),
            make_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.expert3 = PointExpert(
            hidden_channels,
            self.references_per_expert[0],
        )
        self.expert4 = PointExpert(
            hidden_channels,
            self.references_per_expert[1],
        )
        self.expert5 = PointExpert(
            hidden_channels,
            self.references_per_expert[2],
        )

        for expert_index, references in enumerate(
            self.references_per_expert
        ):
            side = math.isqrt(references)
            if side * side != references:
                raise ValueError(
                    "每个 native_multiscale num_references 必须是完全平方数"
                )
            positions = (
                torch.arange(side, dtype=torch.float32) + 0.5
            ) / side
            ref_y, ref_x = torch.meshgrid(
                positions,
                positions,
                indexing="ij",
            )
            reference_offsets = torch.stack(
                [ref_x, ref_y],
                dim=-1,
            ).reshape(references, 2)
            self.register_buffer(
                f"reference_offsets_{expert_index}",
                reference_offsets,
                persistent=False,
            )

    def _forward_level(
        self,
        feature: torch.Tensor,
        projection: nn.Module,
        expert: PointExpert,
        expert_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = projection(feature)
        confidence_logits, offsets = expert(projected)
        batch_size, _, height, width = confidence_logits.shape
        references = self.references_per_expert[expert_index]

        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                height,
                device=projected.device,
                dtype=projected.dtype,
            ),
            torch.arange(
                width,
                device=projected.device,
                dtype=projected.dtype,
            ),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        reference_offsets = getattr(
            self,
            f"reference_offsets_{expert_index}",
        ).to(projected.dtype)
        base_points = (
            grid + reference_offsets
        ) * self.output_strides[expert_index]
        base_points = base_points.reshape(1, -1, 2)

        points = (
            base_points
            + torch.tanh(
                offsets.permute(0, 3, 4, 1, 2).reshape(
                    batch_size,
                    -1,
                    2,
                )
            )
            * self.offset_range
            * self.effective_strides[expert_index]
        )
        logits = confidence_logits.permute(
            0, 2, 3, 1
        ).reshape(batch_size, -1)
        return (
            logits,
            points,
            base_points.expand(batch_size, -1, -1),
            torch.full(
                (batch_size, logits.shape[1]),
                expert_index,
                dtype=torch.long,
                device=logits.device,
            ),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        routing_mode: str = "native",
        expert_index: int | None = None,
    ) -> dict[str, object]:
        valid_modes = {"native", "expert_only"}
        if routing_mode not in valid_modes:
            raise ValueError(
                f"未知 routing_mode={routing_mode!r}；"
                f"可选值为 {sorted(valid_modes)}"
            )
        if routing_mode == "expert_only" and (
            expert_index is None
            or not 0 <= expert_index < self.num_experts
        ):
            raise ValueError(
                "routing_mode='expert_only' 时 expert_index 必须为 0、1 或 2"
            )
        projections = (self.proj3, self.proj4, self.proj5)
        experts = (self.expert3, self.expert4, self.expert5)

        if routing_mode == "expert_only":
            # 只计算被选中的专家，跳过其余两个投影/专家头。
            output = self._forward_level(
                features[expert_index],
                projections[expert_index],
                experts[expert_index],
                expert_index,
            )
            logits, points, base_points, expert_indices = output
            return {
                "architecture": self.architecture,
                "logits": logits,
                "points": points,
                "base_points": base_points,
                "expert_indices": expert_indices,
                "source_expert": expert_indices,
                "expert_logits": (logits,),
                "expert_points": (points,),
                "expert_base_points": (base_points,),
                "references_per_expert": self.references_per_expert,
                "output_strides": self.output_strides,
                "effective_strides": self.effective_strides,
            }

        level_outputs = tuple(
            self._forward_level(
                feature,
                projection,
                expert,
                index,
            )
            for index, (feature, projection, expert) in enumerate(
                zip(features, projections, experts)
            )
        )
        expert_logits = tuple(output[0] for output in level_outputs)
        expert_points = tuple(output[1] for output in level_outputs)
        expert_base_points = tuple(
            output[2] for output in level_outputs
        )

        logits = torch.cat(expert_logits, dim=1)
        points = torch.cat(expert_points, dim=1)
        base_points = torch.cat(expert_base_points, dim=1)
        expert_indices = torch.cat(
            [output[3] for output in level_outputs],
            dim=1,
        )

        if routing_mode == "expert_only":
            keep = expert_indices == expert_index
            logits = logits[keep].reshape(
                logits.shape[0],
                -1,
            )
            points = points[keep].reshape(
                points.shape[0],
                -1,
                2,
            )
            base_points = base_points[keep].reshape(
                base_points.shape[0],
                -1,
                2,
            )
            expert_indices = expert_indices[keep].reshape(
                expert_indices.shape[0],
                -1,
            )

        return {
            "architecture": self.architecture,
            "logits": logits,
            "points": points,
            "base_points": base_points,
            "expert_indices": expert_indices,
            "source_expert": expert_indices,
            "expert_logits": expert_logits,
            "expert_points": expert_points,
            "expert_base_points": expert_base_points,
            "references_per_expert": self.references_per_expert,
            "output_strides": self.output_strides,
            "effective_strides": self.effective_strides,
        }
