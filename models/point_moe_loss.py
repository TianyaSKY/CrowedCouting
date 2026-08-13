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
    """点级 Scale-MoE Head 的损失函数。

    总损失：
        L = L_cls + 5 * L_point + 0.05 * L_count + 0.01 * L_balance

    匹配流程（对 Router 处理后的最终候选点）：
        网络输出所有最终候选点
            -> 计算候选点与真实点之间的代价
            -> 匈牙利一对一匹配
            -> 匹配候选点为正样本，未匹配候选点为负样本
    """

    def __init__(
        self,
        coordinate_weight: float = 5.0,
        count_weight: float = 0.05,
        balance_weight: float = 0.01,
    ) -> None:
        super().__init__()

        self.coordinate_weight = coordinate_weight
        self.count_weight = count_weight
        self.balance_weight = balance_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth_points: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = predictions["logits"]
        points = predictions["points"]
        gates = predictions["gates"]

        batch_size = logits.shape[0]
        image_height, image_width = image_size

        scale = points.new_tensor(
            [float(image_width), float(image_height)]
        )

        cls_loss = logits.new_zeros(())
        point_loss = logits.new_zeros(())
        count_loss = logits.new_zeros(())

        positive_gates = []

        for batch_index in range(batch_size):
            gt = ground_truth_points[batch_index].to(
                points.device
            )

            pred_logits = logits[batch_index]
            pred_points = points[batch_index]

            number_of_gt = gt.shape[0]

            targets = torch.zeros_like(pred_logits)

            if number_of_gt > 0:
                if pred_points.shape[0] < number_of_gt:
                    raise RuntimeError(
                        "候选点数量小于真实点数量"
                    )

                normalized_gt = gt / scale
                normalized_pred = pred_points / scale

                coordinate_cost = torch.cdist(
                    normalized_gt,
                    normalized_pred,
                    p=1,
                )

                confidence_cost = (
                    -pred_logits.sigmoid().unsqueeze(0)
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

                targets[pred_indices] = 1.0

                point_loss = point_loss + (
                    F.smooth_l1_loss(
                        normalized_pred[pred_indices],
                        normalized_gt[gt_indices],
                        reduction="sum",
                    )
                    / number_of_gt
                )

                positive_gates.append(
                    gates[batch_index, pred_indices]
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

        if positive_gates:
            positive_gates_tensor = torch.cat(
                positive_gates,
                dim=0,
            )

            expert_usage = positive_gates_tensor.mean(
                dim=0
            )

            target_usage = torch.full_like(
                expert_usage,
                1.0 / expert_usage.numel(),
            )

            balance_loss = (
                expert_usage - target_usage
            ).pow(2).sum()
        else:
            balance_loss = logits.new_zeros(())

        total_loss = (
            cls_loss
            + self.coordinate_weight * point_loss
            + self.count_weight * count_loss
            + self.balance_weight * balance_loss
        )

        return total_loss, {
            "cls": cls_loss.detach(),
            "point": point_loss.detach(),
            "count": count_loss.detach(),
            "balance": balance_loss.detach(),
        }
