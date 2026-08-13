from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    输入 P3、P4、P5，输出统一的候选点集合。

    三个专家共享 P3 的定位细节、感受野不同：
        Expert3 = E3(F3)
        Expert4 = E4(F3 + Up(F4))
        Expert5 = E5(F3 + Up(F5))

    Router 以 Concat(P3, Up(P4), Up(P5)) 为输入，
    为每个候选点（P3 网格 x K 个参考点）输出 [g3, g4, g5]。

    输出：
        logits: [B, Q]          每个候选点的前景 logit
        points: [B, Q, 2]       解码后的人员坐标（像素）
        gates:  [B, Q, 3]       每个候选点的路由权重
        base_points: [B, Q, 2]  未加偏移的参考点坐标（像素）
        route_logits: [B, K, 3, H, W]  原始路由 logits（未归一化）
    """

    def __init__(
        self,
        feature_channels: tuple[int, int, int],
        hidden_channels: int = 128,
        num_references: int = 4,
        output_stride: int = 8,
        offset_range: float = 2.0,
    ) -> None:
        super().__init__()

        if num_references not in {1, 4, 9}:
            raise ValueError(
                "num_references 建议使用 1、4 或 9"
            )

        self.num_references = num_references
        self.num_experts = 3
        self.output_stride = output_stride
        self.offset_range = offset_range

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
            num_references,
        )
        self.expert4 = PointExpert(
            hidden_channels,
            num_references,
        )
        self.expert5 = PointExpert(
            hidden_channels,
            num_references,
        )

        self.router = nn.Sequential(
            ConvBlock(
                hidden_channels * 3,
                hidden_channels,
            ),
            nn.Conv2d(
                hidden_channels,
                num_references * self.num_experts,
                kernel_size=1,
            ),
        )

        # 初始路由 logits 全为 0 -> softmax 后三个专家等权，
        # 避免训练初期路由坍缩
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        side = int(math.sqrt(num_references))
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
        ).reshape(num_references, 2)

        self.register_buffer(
            "reference_offsets",
            reference_offsets,
            persistent=False,
        )

    def forward(
        self,
        features: list[torch.Tensor],
        temperature: float = 1.0,
        hard_route: bool = False,
    ) -> dict[str, torch.Tensor]:
        p3, p4, p5 = features

        f3 = self.proj3(p3)

        f4 = F.interpolate(
            self.proj4(p4),
            size=f3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f5 = F.interpolate(
            self.proj5(p5),
            size=f3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 每个专家都保留 P3 精细定位特征
        score3, offset3 = self.expert3(f3)
        score4, offset4 = self.expert4(f3 + f4)
        score5, offset5 = self.expert5(f3 + f5)

        batch_size, _, height, width = score3.shape

        route_logits = self.router(
            torch.cat([f3, f4, f5], dim=1)
        ).view(
            batch_size,
            self.num_references,
            self.num_experts,
            height,
            width,
        )

        soft_gate = F.softmax(
            route_logits / temperature,
            dim=2,
        )

        if hard_route:
            expert_index = soft_gate.argmax(
                dim=2,
                keepdim=True,
            )

            hard_gate = torch.zeros_like(
                soft_gate
            ).scatter_(
                2,
                expert_index,
                1.0,
            )

            if self.training:
                # Straight-through Top-1：前向走硬路由，梯度经软路由回传
                gate = (
                    hard_gate
                    + soft_gate
                    - soft_gate.detach()
                )
            else:
                gate = hard_gate
        else:
            gate = soft_gate

        expert_scores = torch.stack(
            [score3, score4, score5],
            dim=2,
        )

        expert_offsets = torch.stack(
            [offset3, offset4, offset5],
            dim=2,
        )

        final_logits = (
            gate * expert_scores
        ).sum(dim=2)

        final_offsets = (
            gate.unsqueeze(3) * expert_offsets
        ).sum(dim=2)

        # 展平顺序：H、W、K
        final_logits = final_logits.permute(
            0, 2, 3, 1
        ).reshape(batch_size, -1)

        final_offsets = final_offsets.permute(
            0, 3, 4, 1, 2
        ).reshape(batch_size, -1, 2)

        final_gates = gate.permute(
            0, 3, 4, 1, 2
        ).reshape(
            batch_size,
            -1,
            self.num_experts,
        )

        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                height,
                device=f3.device,
                dtype=f3.dtype,
            ),
            torch.arange(
                width,
                device=f3.device,
                dtype=f3.dtype,
            ),
            indexing="ij",
        )

        grid = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).unsqueeze(2)

        reference_points = (
            grid
            + self.reference_offsets.to(f3.dtype)
        ) * self.output_stride

        reference_points = reference_points.reshape(
            1,
            -1,
            2,
        )

        points = (
            reference_points
            + torch.tanh(final_offsets)
            * self.offset_range
            * self.output_stride
        )

        return {
            "logits": final_logits,
            "points": points,
            "gates": final_gates,
            "base_points": reference_points.expand(
                batch_size,
                -1,
                -1,
            ),
            "route_logits": route_logits,
        }
