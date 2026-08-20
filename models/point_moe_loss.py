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
    """Native multiscale classification, localization, and count loss.

    Warmup uses independent per-expert matching.  Competitive training and
    validation use an expert-balanced preselection followed by one global
    Hungarian assignment:

        L = L_cls + 5 * L_point + 0.05 * L_count
    """

    def __init__(
        self,
        coordinate_weight: float = 5.0,
        count_weight: float = 0.05,
        match_top_k: int = 2000,
        match_position_weight: float = 5.0,
        match_confidence_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if match_top_k <= 0:
            raise ValueError("match_top_k 必须为正数")
        if match_position_weight <= 0:
            raise ValueError("match_position_weight 必须为正数")
        if match_confidence_weight < 0:
            raise ValueError("match_confidence_weight 不能为负数")
        self.coordinate_weight = coordinate_weight
        self.count_weight = count_weight
        self.match_top_k = match_top_k
        self.match_position_weight = match_position_weight
        self.match_confidence_weight = match_confidence_weight


    @staticmethod
    def _native_match_candidates(
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt: torch.Tensor,
        scale: torch.Tensor,
        match_top_k: int,
        match_position_weight: float,
        match_confidence_weight: float,
        expert_indices: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Preselect candidates, then return differentiable Hungarian matches."""
        number_of_gt = gt.shape[0]
        if number_of_gt == 0:
            empty = torch.empty(
                0,
                dtype=torch.long,
                device=pred_logits.device,
            )
            return empty, empty, pred_points[:0], gt[:0]

        top_k = max(
            match_top_k,
            number_of_gt,
            3 if expert_indices is not None else 1,
        )
        if pred_logits.shape[0] > top_k:
            if expert_indices is None:
                match_indices = pred_logits.topk(top_k).indices
            elif expert_indices.unique().numel() == 1:
                # 单专家消融（expert_only）：整个 top_k 预算归该专家
                match_indices = pred_logits.topk(top_k).indices
            else:
                quotas = [
                    top_k // 3
                    + int(expert_index < top_k % 3)
                    for expert_index in range(3)
                ]
                selected_indices = []
                for expert_index, quota in enumerate(quotas):
                    if quota <= 0:
                        continue
                    expert_candidates = torch.nonzero(
                        expert_indices == expert_index,
                        as_tuple=False,
                    ).flatten()
                    if expert_candidates.numel() > quota:
                        selected_local = pred_logits[
                            expert_candidates
                        ].topk(quota).indices
                        expert_candidates = expert_candidates[
                            selected_local
                        ]
                    selected_indices.append(expert_candidates)
                match_indices = torch.cat(selected_indices)
        else:
            match_indices = None

        if match_indices is not None:
            match_logits = pred_logits[match_indices]
            match_points = pred_points[match_indices]
        else:
            match_logits = pred_logits
            match_points = pred_points

        if match_points.shape[0] < number_of_gt:
            raise RuntimeError("候选点数量小于真实点数量")

        normalized_gt = gt / scale
        normalized_pred = match_points / scale
        coordinate_cost = torch.cdist(
            normalized_gt,
            normalized_pred,
            p=1,
        )
        total_cost = (
            match_position_weight * coordinate_cost
            - match_confidence_weight
            * match_logits.sigmoid().unsqueeze(0)
        )
        gt_indices, pred_indices = linear_sum_assignment(
            total_cost.detach().float().cpu().numpy()
        )
        gt_indices = torch.as_tensor(
            gt_indices,
            dtype=torch.long,
            device=pred_logits.device,
        )
        pred_indices = torch.as_tensor(
            pred_indices,
            dtype=torch.long,
            device=pred_logits.device,
        )
        if match_indices is not None:
            matched_full_indices = match_indices[pred_indices]
        else:
            matched_full_indices = pred_indices
        return (
            matched_full_indices,
            gt_indices,
            pred_points[matched_full_indices],
            gt[gt_indices],
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth_points: list[torch.Tensor],
        image_size: tuple[int, int],
        matching_mode: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if matching_mode not in {"independent", "competitive"}:
            raise ValueError(
                "native_multiscale matching_mode 必须是 "
                "'independent' 或 'competitive'"
            )

        logits = predictions["logits"]
        points = predictions["points"]
        gates = predictions.get("gates")
        if gates is None:
            expert_indices = predictions["expert_indices"].long()
            gates = F.one_hot(
                expert_indices,
                num_classes=3,
            ).to(dtype=logits.dtype)
        else:
            expert_indices = predictions.get(
                "expert_indices",
                gates.argmax(dim=-1),
            ).long()

        batch_size = logits.shape[0]
        image_height, image_width = image_size
        scale = points.new_tensor(
            [float(image_width), float(image_height)]
        )
        num_experts = 3

        cls_loss = logits.new_zeros(())
        point_loss = logits.new_zeros(())
        count_loss = logits.new_zeros(())
        winner_hist = logits.new_zeros(num_experts)
        positive_count = logits.new_zeros(num_experts)
        matched_distance_sum = logits.new_zeros(num_experts)
        matched_confidence_sum = logits.new_zeros(num_experts)
        matched_points_total = 0

        for batch_index in range(batch_size):
            gt = ground_truth_points[batch_index].to(
                logits.device
            )
            number_of_gt = gt.shape[0]
            image_cls_loss = logits.new_zeros(())
            image_point_loss = logits.new_zeros(())
            image_count_loss = logits.new_zeros(())

            if matching_mode == "independent":
                for expert_index in range(num_experts):
                    expert_mask = (
                        expert_indices[batch_index]
                        == expert_index
                    )
                    expert_logits = logits[batch_index][
                        expert_mask
                    ]
                    expert_points = points[batch_index][
                        expert_mask
                    ]
                    targets = torch.zeros_like(expert_logits)
                    if number_of_gt > 0:
                        (
                            matched_indices,
                            gt_indices,
                            matched_points,
                            matched_gt,
                        ) = self._native_match_candidates(
                            expert_logits,
                            expert_points,
                            gt,
                            scale,
                            self.match_top_k,
                            self.match_position_weight,
                            self.match_confidence_weight,
                        )
                        targets[matched_indices] = 1.0
                        image_point_loss = (
                            image_point_loss
                            + F.smooth_l1_loss(
                                matched_points / scale,
                                matched_gt / scale,
                                reduction="sum",
                                beta=0.02,
                            )
                            / number_of_gt
                        )
                        matched_distance_sum[expert_index] += (
                            torch.abs(
                                matched_points - matched_gt
                            )
                            .sum(dim=-1)
                            .detach()
                            .sum()
                        )
                        matched_confidence_sum[
                            expert_index
                        ] += (
                            expert_logits[matched_indices]
                            .sigmoid()
                            .detach()
                            .sum()
                        )
                        positive_count[expert_index] += (
                            matched_indices.numel()
                        )
                        matched_points_total += (
                            matched_indices.numel()
                        )

                    image_cls_loss = image_cls_loss + (
                        sigmoid_focal_loss(
                            expert_logits,
                            targets,
                        ).sum()
                        / max(number_of_gt, 1)
                    )
                    image_count_loss = image_count_loss + (
                        torch.abs(
                            expert_logits.sigmoid().sum()
                            - float(number_of_gt)
                        )
                        / (number_of_gt + 1.0)
                    )

                image_cls_loss = image_cls_loss / num_experts
                image_point_loss = image_point_loss / num_experts
                image_count_loss = image_count_loss / num_experts
            else:
                targets = torch.zeros_like(logits[batch_index])
                if number_of_gt > 0:
                    (
                        matched_indices,
                        gt_indices,
                        matched_points,
                        matched_gt,
                    ) = self._native_match_candidates(
                        logits[batch_index],
                        points[batch_index],
                        gt,
                        scale,
                        self.match_top_k,
                        self.match_position_weight,
                        self.match_confidence_weight,
                        expert_indices[batch_index],
                    )
                    targets[matched_indices] = 1.0
                    matched_sources = expert_indices[
                        batch_index
                    ][matched_indices]
                    winner_hist += F.one_hot(
                        matched_sources,
                        num_classes=num_experts,
                    ).to(dtype=logits.dtype).sum(dim=0)
                    positive_count += F.one_hot(
                        matched_sources,
                        num_classes=num_experts,
                    ).to(dtype=logits.dtype).sum(dim=0)
                    matched_distance_sum += (
                        torch.zeros_like(matched_distance_sum)
                        .index_add(
                            0,
                            matched_sources,
                            torch.abs(
                                matched_points - matched_gt
                            )
                            .sum(dim=-1)
                            .detach(),
                        )
                    )
                    matched_confidence_sum += (
                        torch.zeros_like(matched_confidence_sum)
                        .index_add(
                            0,
                            matched_sources,
                            logits[batch_index][
                                matched_indices
                            ]
                            .sigmoid()
                            .detach(),
                        )
                    )
                    matched_points_total += matched_indices.numel()
                    image_point_loss = F.smooth_l1_loss(
                        matched_points / scale,
                        matched_gt / scale,
                        reduction="sum",
                        beta=0.02,
                    ) / number_of_gt

                image_cls_loss = (
                    sigmoid_focal_loss(
                        logits[batch_index],
                        targets,
                    ).sum()
                    / max(number_of_gt, 1)
                )
                image_count_loss = torch.abs(
                    logits[batch_index].sigmoid().sum()
                    - float(number_of_gt)
                ) / (number_of_gt + 1.0)

            cls_loss += image_cls_loss
            point_loss += image_point_loss
            count_loss += image_count_loss

        cls_loss = cls_loss / batch_size
        point_loss = point_loss / batch_size
        count_loss = count_loss / batch_size
        total_loss = (
            cls_loss
            + self.coordinate_weight * point_loss
            + self.count_weight * count_loss
        )

        matched_distance_mean = (
            matched_distance_sum
            / positive_count.clamp_min(1.0)
        )
        matched_confidence_mean = (
            matched_confidence_sum
            / positive_count.clamp_min(1.0)
        )
        return total_loss, {
            "cls": cls_loss.detach(),
            "point": point_loss.detach(),
            "count": count_loss.detach(),
            "matching_mode": matching_mode,
            "winner_hist": winner_hist.detach(),
            "matched_expert_hist": winner_hist.detach(),
            "positive_count": positive_count.detach(),
            "matched_distance_sum": matched_distance_sum.detach(),
            "matched_distance_mean": (
                matched_distance_mean.detach()
            ),
            "matched_confidence_sum": (
                matched_confidence_sum.detach()
            ),
            "matched_confidence_mean": (
                matched_confidence_mean.detach()
            ),
            "matched_count": logits.new_tensor(
                matched_points_total,
                dtype=logits.dtype,
            ),
        }

