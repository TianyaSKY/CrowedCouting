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
        L = L_cls + 5 * L_point + 0.05 * L_count + 0.05 * L_route

    匹配流程（对 Router 处理后的最终候选点）：
        网络输出所有最终候选点
            -> 按置信度取 top-K 作为匹配候选（K = max(match_top_k, n_gt)，
               避免 n_gt x Q(25600) 的匈牙利指派在密集裁剪上卡死）
            -> 计算候选点与真实点之间的代价
            -> 匈牙利一对一匹配
            -> 匹配候选点为正样本，未匹配候选点为负样本

    L_route 为尺度路由监督：对每个匹配正样本，用 GT 点的最近邻间距
    （knn_k=1）估计局部尺度，映射为三专家软目标（小间距->精细 E3，
    大间距->大范围 E5），与 Router logits 做交叉熵。替代原先强制 1/3
    均匀使用的 balance loss，让 Router 学到数据驱动的尺度语义。
    """

    def __init__(
        self,
        coordinate_weight: float = 5.0,
        count_weight: float = 0.05,
        route_weight: float = 0.05,
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
        self.route_weight = route_weight
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
        """GT 像素坐标 -> 三专家软目标。

        用第 knn_k 近邻间距（默认 1，即最近邻间距）估计每个 GT 点的
        局部尺度，在对数尺度空间做高斯映射：间距小（密集人群）-> E3
        精细，间距大（稀疏/大目标）-> E5 大范围。返回 [N, 3] 的
        归一化软目标。
        """
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
        route_logits = predictions["route_logits"]

        batch_size = logits.shape[0]
        image_height, image_width = image_size

        num_experts = route_logits.shape[2]

        # 展平顺序与 head 中 final_gates 一致: [B, H, W, K, E] -> [B, Q, E]
        route_logits_flat = route_logits.permute(
            0, 3, 4, 1, 2
        ).reshape(batch_size, -1, num_experts)

        scale = points.new_tensor(
            [float(image_width), float(image_height)]
        )

        cls_loss = logits.new_zeros(())
        point_loss = logits.new_zeros(())
        count_loss = logits.new_zeros(())
        route_loss = logits.new_zeros(())

        # 诊断：GT 尺度目标的 argmax 分布，与预测 gate 分布对比
        target_gate_hist = logits.new_zeros(num_experts)
        target_gate_points = 0

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
                # 否则 n_gt x Q(25600) 的线性指派在密集裁剪上会卡死
                # （实测单张 1407 个 GT 的裁剪可挂住数分钟）。
                # K = max(match_top_k, n_gt) 保证每个 GT 仍能匹配到一个候选。
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
                        # 归一化坐标下典型定位误差 ~0.005(≈2px/384)，
                        # 默认 beta=1.0 使所有误差落入二次区，损失被压到
                        # ~1e-5 量级、定位分支几乎没有梯度；
                        # beta=0.02(≈8px) 让 2-8px 误差走线性/准线性区
                        beta=0.02,
                    )
                    / number_of_gt
                )

                # 尺度路由监督：GT 局部最近邻间距 -> 专家软目标 -> Router CE。
                # 间距小(密集人群) -> 精细 E3，间距大(稀疏/大目标) -> 大范围 E5。
                # 需要至少 knn_k+1 个点才能计算第 knn_k 近邻。
                if number_of_gt >= self.knn_k + 1:
                    target_gate = self.scale_targets(
                        gt,
                        self.knn_k,
                        self.scale_centers,
                        self.scale_sigma_octaves,
                    )

                    target_gate_hist += target_gate.argmax(
                        dim=1
                    ).bincount(minlength=num_experts).float()
                    target_gate_points += number_of_gt

                    matched_route = route_logits_flat[
                        batch_index
                    ][matched_full_indices]

                    # macro 类别平衡：GT 目标 E0~60%/E1~27%/E2~13%，
                    # 逐点 CE 会让 E0 天然主导 route 梯度，E2 永远学不动。
                    # 改为：按 hard 目标类别分组，组内 soft-target CE 取
                    # 均值，再对三个类别取均值——各类别对 Router 训练
                    # 同等重要，但不强制最终路由比例 33/33/33。
                    per_point_route = -(
                        target_gate[gt_indices]
                        * F.log_softmax(
                            matched_route, dim=-1
                        )
                    ).sum(dim=-1)

                    hard_target = target_gate[
                        gt_indices
                    ].argmax(dim=-1)

                    class_route_losses = []
                    for expert_id in range(num_experts):
                        mask = hard_target == expert_id
                        if mask.any():
                            class_route_losses.append(
                                per_point_route[mask].mean()
                            )

                    if class_route_losses:
                        route_loss = route_loss + torch.stack(
                            class_route_losses
                        ).mean()

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
        route_loss = route_loss / batch_size

        total_loss = (
            cls_loss
            + self.coordinate_weight * point_loss
            + self.count_weight * count_loss
            + self.route_weight * route_loss
        )

        return total_loss, {
            "cls": cls_loss.detach(),
            "point": point_loss.detach(),
            "count": count_loss.detach(),
            "route": route_loss.detach(),
            "gate_target": (
                target_gate_hist / max(target_gate_points, 1)
            ).detach(),
        }
