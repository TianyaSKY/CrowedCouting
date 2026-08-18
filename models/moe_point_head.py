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

    三个专家对应不同尺度（多感受野），全部在 P3 网格上预测：
        Expert3 = E3(F3)
        Expert4 = E4(Up(F4) + alpha4 * F3)
        Expert5 = E5(Up(F5) + alpha5 * F3)

    alpha4/alpha5 为可学习横向融合系数，初始为 0：专家先只吃自身尺度特征
    （E3 精细 / E4 中层 / E5 大范围），需要时再学习引入 P3 细节，
    避免三个专家共享完整 P3 而退化成近似副本。

    Router 以 Concat(P3, Up(P4), Up(P5)) 为输入，
    为每个候选点（P3 网格 x K 个参考点）输出 [g3, g4, g5]。

    输出：
        logits: [B, Q]          每个候选点的前景 logit
        points: [B, Q, 2]       解码后的人员坐标（像素）
        gates:  [B, Q, 3]       当前 forward 实际使用的路由权重
        base_points: [B, Q, 2]  未加偏移的参考点坐标（像素）
        route_logits: [B, K, 3, H, W]  原始路由 logits（未归一化）
        route_probabilities: [B, K, 3, H, W] 完整三专家 softmax，
            不受 Drop-1 或验证模式影响
        dropped_expert: [B, Q] 训练 Drop-1 的候选级被丢弃专家
        active_expert_mask: [B, Q, 3] 当前 forward 中活跃的专家

    ``routing_mode`` 支持：
        ``train_drop1``: 训练时每个候选随机丢弃一个专家；
        ``full3_soft``: 验证用完整三专家 softmax；
        ``top2``: 验证用确定性 Soft Top-2；
        ``top1``: 验证诊断用确定性 Top-1；
        ``expert_only``: 诊断时只启用 ``expert_index``。
    """


    def __init__(
        self,
        feature_channels: tuple[int, int, int],
        hidden_channels: int = 256,
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

        # 可学习横向融合系数：初始 0（纯尺度输入），需要时可学习引入 P3 细节
        self.alpha4 = nn.Parameter(torch.zeros(1))
        self.alpha5 = nn.Parameter(torch.zeros(1))

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

    def _drop_one_expert(
        self,
        route_logits: torch.Tensor,
        temperature: float,
        router_grad: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return candidate-level random Drop-1 gates and masks."""
        batch_size, _, _, height, width = route_logits.shape
        dropped_expert = torch.randint(
            self.num_experts,
            (batch_size, self.num_references, height, width),
            device=route_logits.device,
        )
        dropped_mask = F.one_hot(
            dropped_expert,
            num_classes=self.num_experts,
        ).permute(0, 1, 4, 2, 3).to(dtype=torch.bool)
        if router_grad:
            masked_logits = route_logits.masked_fill(
                dropped_mask,
                float("-inf"),
            )
            gate = F.softmax(
                masked_logits / temperature,
                dim=2,
            )
            active_mask = ~dropped_mask
            gate = gate.masked_fill(~active_mask, 0.0)
            gate = gate + (
                active_mask.to(dtype=gate.dtype)
                * torch.finfo(gate.dtype).tiny
            )
            gate = gate / gate.sum(dim=2, keepdim=True)
        else:
            gate = (
                (~dropped_mask).to(dtype=route_logits.dtype)
                / float(self.num_experts - 1)
            )
        return gate, dropped_expert, dropped_mask

    @staticmethod
    def _deterministic_top2_gate(
        soft_gate: torch.Tensor,
    ) -> torch.Tensor:
        active_indices = soft_gate.topk(
            k=2,
            dim=2,
        ).indices
        active_mask = torch.zeros_like(
            soft_gate,
            dtype=torch.bool,
        ).scatter_(
            2,
            active_indices,
            True,
        )
        gate = soft_gate.masked_fill(~active_mask, 0.0)
        gate = gate + (
            active_mask.to(dtype=gate.dtype)
            * torch.finfo(gate.dtype).tiny
        )
        return gate / gate.sum(dim=2, keepdim=True)

    def forward(
        self,
        features: list[torch.Tensor],
        temperature: float = 1.0,
        routing_mode: str = "full3_soft",
        router_grad: bool = True,
        expert_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        p3, p4, p5 = features
        if temperature <= 0:
            raise ValueError("temperature 必须为正数")
        valid_modes = {
            "train_drop1",
            "full3_soft",
            "top2",
            "top1",
            "expert_only",
        }
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

        # 每个专家只接收自身尺度的特征（E3 精细 / E4 中层 / E5 大范围），
        # 横向融合 alpha*F3 初始为 0，防止专家输入同质化。
        score3, offset3 = self.expert3(f3)
        score4, offset4 = self.expert4(f4 + self.alpha4 * f3)
        score5, offset5 = self.expert5(f5 + self.alpha5 * f3)

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

        # 始终保留完整三专家概率，供 task-only 诊断使用。
        route_probabilities = F.softmax(
            route_logits / temperature,
            dim=2,
        )

        dropped_expert = torch.full(
            (
                batch_size,
                self.num_references,
                height,
                width,
            ),
            -1,
            dtype=torch.long,
            device=route_logits.device,
        )
        dropped_mask = torch.zeros_like(
            route_logits,
            dtype=torch.bool,
        )

        if routing_mode == "train_drop1":
            gate, dropped_expert, dropped_mask = (
                self._drop_one_expert(
                    route_logits,
                    temperature,
                    router_grad,
                )
            )
        elif routing_mode == "full3_soft":
            gate = route_probabilities
        elif routing_mode == "top2":
            gate = self._deterministic_top2_gate(
                route_probabilities
            )
        elif routing_mode == "top1":
            expert_index_tensor = route_probabilities.argmax(
                dim=2,
                keepdim=True,
            )
            gate = torch.zeros_like(route_probabilities).scatter_(
                2,
                expert_index_tensor,
                1.0,
            )
        else:
            gate = torch.zeros_like(route_probabilities).scatter_(
                2,
                torch.full(
                    (
                        batch_size,
                        self.num_references,
                        1,
                        height,
                        width,
                    ),
                    expert_index,
                    dtype=torch.long,
                    device=route_logits.device,
                ),
                1.0,
            )

        expert_scores = torch.stack(
            [score3, score4, score5],
            dim=2,
        )

        expert_offsets = torch.stack(
            [offset3, offset4, offset5],
            dim=2,
        )

        # soft diagnostic 在概率空间混合: p = sum(g * sigmoid(z))，再转回 logit，
        # 避免 logits 线性混合时专家置信度相互抵消。
        expert_probabilities = expert_scores.sigmoid()
        mixed_probability = (
            gate * expert_probabilities
        ).sum(dim=2).clamp(1e-7, 1.0 - 1e-7)
        final_logits = torch.log(
            mixed_probability
        ) - torch.log1p(-mixed_probability)

        final_offsets = (
            gate.unsqueeze(3) * expert_offsets
        ).sum(dim=2)

        # 展平顺序：H、W、K。
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
        final_dropped_expert = dropped_expert.permute(
            0, 2, 3, 1
        ).reshape(batch_size, -1)
        final_active_expert_mask = (
            (~dropped_mask)
            if routing_mode == "train_drop1"
            else gate > 0
        ).permute(
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
            "route_probabilities": route_probabilities,
            "dropped_expert": final_dropped_expert,
            "active_expert_mask": final_active_expert_mask,
        }
