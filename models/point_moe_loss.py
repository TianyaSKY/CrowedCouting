from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = logits.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = (
        probability * targets
        + (1.0 - probability) * (1.0 - targets)
    )

    modulating = (1.0 - p_t).pow(gamma)

    alpha_t = (
        alpha * targets
        + (1.0 - alpha) * (1.0 - targets)
    )

    return alpha_t * modulating * ce_loss


class PointMoELoss(nn.Module):
    """点级 MoE Head 的 task-only 损失函数。

    训练目标只包含分类、定位和计数：

        L = L_cls + 5 * L_point + 0.05 * L_count

    ``scale_targets`` 保留给可选的尺度路由诊断，但不参与训练。
    """

    def __init__(
        self,
        coordinate_weight: float = 5.0,
        count_weight: float = 0.05,
        knn_k: int = 1,
        scale_centers: tuple[float, float, float] = (
            10.0,
            20.0,
            40.0,
        ),
        scale_sigma_octaves: float = 0.6,
        match_top_k: int = 2000,
    ) -> None:
        super().__init__()

        self.coordinate_weight = coordinate_weight
        self.count_weight = count_weight
        self.knn_k = knn_k
        self.scale_centers = scale_centers
        self.scale_sigma_octaves = scale_sigma_octaves
        self.match_top_k = match_top_k

    @staticmethod
    def scale_targets(
        gt: torch.Tensor,
        knn_k: int,
        scale_centers: tuple[float, float, float],
        scale_sigma_octaves: float,
    ) -> torch.Tensor:
        """GT 像素坐标 -> 三专家软目标（仅诊断使用）。"""
        gt_distances = torch.cdist(gt, gt)

        kth_distances = gt_distances.sort(
            dim=1
        ).values[:, knn_k]

        log_scale = torch.log2(
            kth_distances.clamp_min(1.0)
        )

        log_centers = torch.log2(
            gt.new_tensor(
                scale_centers,
                dtype=gt.dtype,
            )
        )

        squared_octaves = (
            (
                log_scale[:, None]
                - log_centers[None, :]
            )
            / scale_sigma_octaves
        ).pow(2)

        return F.softmax(
            -0.5 * squared_octaves,
            dim=1,
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth_points: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = predictions["logits"]
        points = predictions["points"]
        gates = predictions["gates"]
        route_probabilities = predictions.get(
            "route_probabilities"
        )
        if route_probabilities is None:
            route_probabilities = F.softmax(
                predictions["route_logits"],
                dim=2,
            )
        route_probabilities = route_probabilities.permute(
            0, 3, 4, 1, 2
        ).reshape_as(gates)
        batch_size = logits.shape[0]
        image_height, image_width = image_size
        num_experts = gates.shape[2]

        scale = points.new_tensor(
            [float(image_width), float(image_height)]
        )

        cls_loss = logits.new_zeros(())
        point_loss = logits.new_zeros(())
        count_loss = logits.new_zeros(())

        # 只统计 Hungarian 匹配到的正样本，避免背景候选污染
        # Router 分布诊断。这里不对 route logits 计算独立
        # 监督，也不产生任何 Router 监督项。
        matched_probability_sum = logits.new_zeros(
            num_experts
        )
        matched_top1_hist = logits.new_zeros(
            num_experts
        )
        matched_gate_points = 0

        for batch_index in range(batch_size):
            gt = ground_truth_points[batch_index].to(
                points.device
            )

            pred_logits = logits[batch_index]
            pred_points = points[batch_index]
            number_of_gt = gt.shape[0]

            targets = torch.zeros_like(pred_logits)

            if number_of_gt > 0:
                # 匹配候选：只对置信度最高的 top-K 做匈牙利匹配，
                # 否则 n_gt x Q 的线性指派在密集裁剪上会卡死。
                match_logits = pred_logits
                match_points = pred_points
                match_indices = None
                top_k = max(self.match_top_k, number_of_gt)

                if pred_logits.shape[0] > top_k:
                    match_indices = pred_logits.topk(
                        top_k
                    ).indices
                    match_logits = pred_logits[
                        match_indices
                    ]
                    match_points = pred_points[
                        match_indices
                    ]

                if match_points.shape[0] < number_of_gt:
                    raise RuntimeError(
                        "候选点数量小于真实点数量"
                    )

                normalized_gt = gt / scale
                normalized_pred = match_points / scale

                coordinate_cost = torch.cdist(
                    normalized_gt,
                    normalized_pred,
                    p=1,
                )
                confidence_cost = (
                    -match_logits.sigmoid().unsqueeze(0)
                )
                total_cost = (
                    5.0 * coordinate_cost
                    + confidence_cost
                )

                gt_indices, pred_indices = (
                    linear_sum_assignment(
                        total_cost.detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                )

                gt_indices = torch.as_tensor(
                    gt_indices,
                    dtype=torch.long,
                    device=points.device,
                )
                pred_indices = torch.as_tensor(
                    pred_indices,
                    dtype=torch.long,
                    device=points.device,
                )

                if match_indices is not None:
                    matched_full_indices = match_indices[
                        pred_indices
                    ]
                else:
                    matched_full_indices = pred_indices

                targets[matched_full_indices] = 1.0

                point_loss = point_loss + (
                    F.smooth_l1_loss(
                        normalized_pred[pred_indices],
                        normalized_gt[gt_indices],
                        reduction="sum",
                        beta=0.02,
                    )
                    / number_of_gt
                )

                matched_gates = gates[batch_index][
                    matched_full_indices
                ].detach()
                matched_probabilities = route_probabilities[
                    batch_index
                ][matched_full_indices].detach()
                matched_probability_sum += (
                    matched_probabilities.sum(dim=0)
                )
                matched_top1_hist += (
                    matched_gates.argmax(dim=-1)
                    .bincount(minlength=num_experts)
                    .to(dtype=logits.dtype)
                )
                matched_gate_points += int(
                    matched_full_indices.numel()
                )

            cls_loss = cls_loss + (
                sigmoid_focal_loss(
                    pred_logits,
                    targets,
                ).sum()
                / max(number_of_gt, 1)
            )

            soft_count = pred_logits.sigmoid().sum()
            count_loss = count_loss + (
                torch.abs(
                    soft_count - float(number_of_gt)
                )
                / (number_of_gt + 1.0)
            )

        cls_loss = cls_loss / batch_size
        point_loss = point_loss / batch_size
        count_loss = count_loss / batch_size

        total_loss = (
            cls_loss
            + self.coordinate_weight * point_loss
            + self.count_weight * count_loss
        )

        return total_loss, {
            "cls": cls_loss.detach(),
            "point": point_loss.detach(),
            "count": count_loss.detach(),
            "matched_probability_sum": (
                matched_probability_sum.detach()
            ),
            "matched_top1_hist": (
                matched_top1_hist.detach()
            ),
            "matched_gate_count": logits.new_tensor(
                matched_gate_points,
                dtype=logits.dtype,
            ),
        }
