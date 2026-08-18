import argparse
import logging
import os
import shutil
import time

# 解冻 YOLO 时 backward 图骤增，缓存分配器在 Blackwell(RTX50) 等新卡上
# 容易碎片化，导致 cublasCreate 分配失败崩溃；启用可扩展段规避。
# 必须在首次 CUDA 调用前设置。
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo11_moe_point import YOLO11MoEPoint
from models.point_moe_loss import PointMoELoss
from scripts.data.point_dataset import (
    PointDataset,
    point_collate_fn,
)


def setup_logging(log_path: str) -> None:
    """INFO 及以上同时写入控制台与日志文件。

    tqdm 进度条走 stderr，不会混入日志文件。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def validate_cuda_device(device: str) -> None:
    """Fail fast when the installed PyTorch wheel cannot target the GPU."""
    if device != "cuda":
        return

    capability = torch.cuda.get_device_capability()
    target_arch = f"sm_{capability[0] * 10 + capability[1]}"
    supported_arches = tuple(torch.cuda.get_arch_list())
    if supported_arches and target_arch not in supported_arches:
        supported = ", ".join(supported_arches)
        raise RuntimeError(
            "当前 PyTorch CUDA wheel 不支持当前 GPU: "
            f"{torch.cuda.get_device_name()} ({target_arch})。"
            f"torch={torch.__version__}, "
            f"CUDA={torch.version.cuda}, "
            f"支持的架构: {supported}。"
            "请按 requirements.txt 安装支持该 GPU 的 CUDA 版 PyTorch，"
            "例如: python -m pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu130。"
            "不要继续使用当前环境训练。"
        )

    try:
        torch.zeros(1, device="cuda").sum().item()
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA 设备初始化/基础算子探测失败；"
            "请检查 NVIDIA 驱动与 PyTorch CUDA 版本是否匹配。"
        ) from error


def parse_scale_centers(value: str) -> tuple[float, float, float]:
    """Parse and validate the three fixed Router scale centers."""
    centers = tuple(float(item.strip()) for item in value.split(","))
    if len(centers) != 3 or any(center <= 0 for center in centers):
        raise ValueError(
            "--scale-centers 需要三个正数，例如 10,20,40"
        )
    return (centers[0], centers[1], centers[2])


def build_checkpoint_config(
    args,
    criterion: PointMoELoss,
    *,
    temperature: float,
) -> dict[str, object]:
    """Build the canonical task-only settings stored in checkpoints."""
    temperature_schedule = {
        name: getattr(args, name)
        for name in (
            "init_temperature",
            "phase1_temp",
            "soft_temp_floor",
            "min_temperature",
            "temp_floor_epoch",
            "hard_temp_epochs",
        )
    }
    return {
        "crop_size": int(args.crop_size),
        "num_references": int(args.num_references),
        "temperature_schedule": temperature_schedule,
        "temperature": float(temperature),
        # Kept as explicit false values for older evaluation scripts. Unsup-v0
        # never trains or graduates to a hard-routing phase.
        "hard_started": False,
        "hard_route": False,
        "router_training_mode": "task_only",
        "router_warmup_epochs": int(
            args.router_warmup_epochs
        ),
        "diagnose_scale_routing": bool(
            args.diagnose_scale_routing
        ),
        "scale_centers": [
            float(center) for center in criterion.scale_centers
        ],
        "scale_sigma_octaves": float(
            criterion.scale_sigma_octaves
        ),
    }


def build_optimizer(model, args):
    """两个学习率组：YOLO 主干 1e-4，MoE Point Head 1e-3。

    始终包含全部参数（不再按 requires_grad 过滤）：冻结期的参数由优化器
    自动跳过（grad 为 None），解冻后自然开始积累 Adam 状态。整个训练过程
    只创建一次优化器，避免解冻/恢复时重建导致 Head 已积累的动量丢失。
    """
    head_params = list(model.point_head.parameters())
    yolo_params = list(model.yolo.parameters())

    groups = [
        {
            "params": head_params,
            "lr": args.head_lr,
        },
        {
            "params": yolo_params,
            "lr": args.backbone_lr,
        },
    ]

    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )


def _routing_confusion(
    predictions: dict[str, torch.Tensor],
    gt_points: list[torch.Tensor],
    image_size: tuple[int, int],
    device: torch.device,
    match_top_k: int = 2000,
    knn_k: int = 1,
    scale_centers: tuple[float, float, float] = (
        10.0,
        20.0,
        40.0,
    ),
    scale_sigma_octaves: float = 0.6,
) -> tuple[torch.Tensor, int]:
    """单个 batch 的可选尺度路由诊断混淆矩阵。

    匹配口径与 PointMoELoss 完全一致：置信度 top-K
    （K=max(match_top_k, n_gt)）候选上做匈牙利指派，
    代价 = 5*归一化 L1 坐标距离 - 置信度。
    行是 GT 最近邻间距映射的诊断尺度类（E0/E1/E2），
    列是匹配候选在 hard gate 上的 argmax 专家。

    该矩阵只用于观察 Router 是否自然形成尺度相关分工，
    不参与训练、评价选模或 soft/hard 切换。
    """
    from models.point_moe_loss import PointMoELoss

    logits = predictions["logits"]
    points = predictions["points"]
    gates = predictions["gates"]

    batch_size = logits.shape[0]
    image_height, image_width = image_size
    scale = points.new_tensor(
        [float(image_width), float(image_height)]
    )

    confusion = torch.zeros(
        3, 3, dtype=torch.int64, device=device
    )
    matched_points = 0

    for batch_index in range(batch_size):
        gt = gt_points[batch_index].to(device)
        number_of_gt = gt.shape[0]

        # 至少 knn_k+1 个点才能计算第 knn_k 近邻间距。
        if number_of_gt < knn_k + 1:
            continue

        pred_logits = logits[batch_index]
        pred_points = points[batch_index]
        pred_gates = gates[batch_index]

        top_k = max(match_top_k, number_of_gt)
        if pred_logits.shape[0] > top_k:
            match_indices = pred_logits.topk(top_k).indices
            match_logits = pred_logits[match_indices]
            match_points = pred_points[match_indices]
        else:
            match_indices = None
            match_logits = pred_logits
            match_points = pred_points

        normalized_gt = gt / scale
        normalized_pred = match_points / scale
        total_cost = (
            5.0
            * torch.cdist(
                normalized_gt,
                normalized_pred,
                p=1,
            )
            - match_logits.sigmoid().unsqueeze(0)
        )

        gt_indices, pred_indices = linear_sum_assignment(
            total_cost.detach().float().cpu().numpy()
        )
        gt_indices = torch.as_tensor(
            gt_indices,
            dtype=torch.long,
            device=device,
        )
        pred_indices = torch.as_tensor(
            pred_indices,
            dtype=torch.long,
            device=device,
        )

        if match_indices is not None:
            matched_full_indices = match_indices[pred_indices]
        else:
            matched_full_indices = pred_indices

        target_gate = PointMoELoss.scale_targets(
            gt,
            knn_k=knn_k,
            scale_centers=scale_centers,
            scale_sigma_octaves=scale_sigma_octaves,
        )
        target_class = target_gate.argmax(dim=1)
        pred_class = pred_gates[
            matched_full_indices
        ].argmax(dim=1)

        confusion.index_add_(
            0,
            target_class,
            torch.nn.functional.one_hot(
                pred_class,
                num_classes=3,
            ).to(device),
        )
        matched_points += number_of_gt

    return confusion, matched_points


def evaluate_count_mae(
    model,
    val_loader,
    device,
    match_top_k: int = 2000,
    soft_temperature: float = 1.0,
    hard_temperature: float = 0.5,
    knn_k: int = 1,
    scale_centers: tuple[float, float, float] = (
        10.0,
        20.0,
        40.0,
    ),
    scale_sigma_octaves: float = 0.6,
    diagnose_scale_routing: bool = False,
):
    """验证集人数 MAE，同时报告 soft 与 hard 的任务前向。

    训练全程使用 soft；hard 只作为验证指标。尺度混淆矩阵只有在
    ``diagnose_scale_routing`` 开启时才计算，并且始终是 diagnostic only。
    """
    model.eval()

    total_abs_error = {"soft": 0.0, "hard": 0.0}
    total_images = 0
    hard_usage = torch.zeros(
        3, dtype=torch.int64, device=device
    )
    route_confusion = torch.zeros(
        3, 3, dtype=torch.int64, device=device
    )
    matched_points = 0

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc="验证中", leave=False
        ):
            images = batch["img"].to(device)
            gt_points = [
                p.to(device) for p in batch["points"]
            ]
            gt_counts = [
                p.shape[0] for p in gt_points
            ]

            for mode, hard_route in (
                ("soft", False),
                ("hard", True),
            ):
                predictions = model(
                    images,
                    temperature=(
                        soft_temperature
                        if not hard_route
                        else hard_temperature
                    ),
                    hard_route=hard_route,
                )

                scores = predictions["logits"].sigmoid()
                pred_counts = scores.sum(dim=1)

                for i, gt_count in enumerate(gt_counts):
                    total_abs_error[mode] += abs(
                        float(pred_counts[i]) - gt_count
                    )

                if hard_route:
                    # 仅统计置信度 > 0.5 的前景候选，避免背景污染
                    # hard top-1 使用率。
                    fg_mask = scores > 0.5
                    hard_usage += (
                        predictions["gates"][fg_mask]
                        .argmax(dim=-1)
                        .bincount(minlength=3)
                    )

                    if diagnose_scale_routing:
                        (
                            batch_confusion,
                            batch_matched,
                        ) = _routing_confusion(
                            predictions,
                            gt_points,
                            images.shape[-2:],
                            device,
                            match_top_k=match_top_k,
                            knn_k=knn_k,
                            scale_centers=scale_centers,
                            scale_sigma_octaves=(
                                scale_sigma_octaves
                            ),
                        )
                        route_confusion += batch_confusion
                        matched_points += batch_matched

            total_images += len(gt_counts)

    model.train()

    soft_mae = total_abs_error["soft"] / max(
        total_images, 1
    )
    hard_mae = total_abs_error["hard"] / max(
        total_images, 1
    )

    return (
        soft_mae,
        hard_mae,
        hard_usage,
        route_confusion,
        matched_points,
    )


def temperature_for_epoch(
    epoch: int,
    hard_route: bool,
    first_hard_epoch: int | None,
    args,
) -> float:
    """保持 baseline 的 2.0 -> 1.3 -> 1.0 soft 温度曲线。

    Router warm-up 只控制 gate 是否固定均匀，不改变 temperature
    schedule。默认 ``temp_floor_epoch=30`` 时，phase1 在 epoch 15
    到达 ``phase1_temp=1.3``；Unsup-v0 训练不启用 hard 分支。
    """
    if hard_route:
        if first_hard_epoch is None:
            return args.soft_temp_floor
        progress = min(
            epoch - first_hard_epoch,
            args.hard_temp_epochs,
        )
        return max(
            args.min_temperature,
            args.soft_temp_floor
            - (
                args.soft_temp_floor
                - args.min_temperature
            )
            * progress
            / max(args.hard_temp_epochs, 1),
        )

    phase1_epoch = max(
        int(args.temp_floor_epoch) // 2,
        1,
    )
    if epoch <= phase1_epoch:
        progress = epoch / phase1_epoch
        temperature = (
            args.init_temperature
            - (args.init_temperature - args.phase1_temp)
            * progress
        )
    else:
        span = max(
            args.temp_floor_epoch - phase1_epoch,
            1,
        )
        progress = min(
            epoch - phase1_epoch,
            span,
        )
        temperature = (
            args.phase1_temp
            - (args.phase1_temp - args.soft_temp_floor)
            * progress
            / span
        )

    return max(temperature, args.soft_temp_floor)




def timestamped_save_dir(base: str) -> str:
    """为输出目录附加启动时间戳，避免不同 run 互相覆盖。"""
    return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"

def dataset_mean_gt_count(dataset) -> float:
    """Read the validation-set mean GT count for normalized MAE."""
    total_points = 0
    for image_path in dataset.image_paths:
        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]
        point_path = os.path.join(
            dataset.points_dir,
            base_name + ".txt",
        )
        if not os.path.exists(point_path):
            continue
        with open(point_path, encoding="utf-8") as point_file:
            total_points += sum(
                len(line.split()) >= 2
                for line in point_file
            )
    return total_points / max(len(dataset), 1)


def train_moe(args):
    if args.router_warmup_epochs < 0:
        raise ValueError("--router-warmup-epochs 不能为负数")
    if args.force_hard_epoch is not None:
        logging.warning(
            "--force-hard-epoch 已废弃；Unsup-v0 全程 soft，参数被忽略"
        )

    # 未显式指定输出目录时自动加时间戳；--resume 沿用原 run 目录。
    if args.save_dir is None:
        if args.resume:
            args.save_dir = os.path.dirname(
                os.path.abspath(args.resume)
            )
        else:
            args.save_dir = timestamped_save_dir(
                "runs/moe_point"
            )

    os.makedirs(args.save_dir, exist_ok=True)
    setup_logging(os.path.join(args.save_dir, "train.log"))
    logging.info("输出目录: %s", args.save_dir)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"使用设备: {device}")
    validate_cuda_device(device)

    # 覆盖保护：从头训练进入已有 save-dir 时备份旧 best_soft。
    if not args.resume:
        old_best = os.path.join(
            args.save_dir, "best_soft.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_soft_prev.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(f"旧 best_soft.pt 已备份到 {backup}")

    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        num_references=args.num_references,
    ).to(device)

    criterion = PointMoELoss(
        scale_centers=parse_scale_centers(args.scale_centers),
        match_top_k=args.match_top_k,
    )

    train_dataset = PointDataset(
        args.data_root,
        split="train",
        crop_size=args.crop_size,
        augment=True,
    )
    val_dataset = PointDataset(
        args.data_root,
        split="val",
        crop_size=args.crop_size,
        augment=False,
    )
    val_mean_gt_count = dataset_mean_gt_count(val_dataset)

    gt_counts = train_dataset.sample_gt_counts(
        num_samples=100
    )
    if gt_counts.size > 0:
        logging.info(
            f"训练裁剪 GT 统计(采样 {gt_counts.size} 张): "
            f"mean={float(gt_counts.mean()):.1f} "
            f"median={float(np.median(gt_counts)):.0f} "
            f"max={int(gt_counts.max())} "
            f"zero_ratio={float((gt_counts == 0).mean()):.1%}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=point_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=point_collate_fn,
    )

    if args.freeze_epochs > 0:
        logging.info(
            f"前 {args.freeze_epochs} 个 epoch 冻结 YOLO Backbone+Neck"
        )
        for param in model.yolo.parameters():
            param.requires_grad = False

    optimizer = build_optimizer(model, args)
    start_epoch = 0
    best_selection_score = float("inf")
    best_soft_selection_score = float("inf")

    if args.resume and os.path.exists(args.resume):
        logging.info(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        saved_config = checkpoint.get("config", {})
        if (
            not isinstance(saved_config, dict)
            or saved_config.get("router_training_mode")
            != "task_only"
        ):
            raise ValueError(
                "只能从 router_training_mode=task_only 的 checkpoint "
                "恢复，避免混用 supervised Router 权重"
            )

        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as error:
            logging.warning(
                f"checkpoint 参数不完全匹配({error})，"
                "缺失部分使用初始化值"
            )
            model.load_state_dict(
                checkpoint["model"], strict=False
            )
        try:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
        except (ValueError, RuntimeError) as error:
            logging.warning(
                f"优化器状态不兼容({error})，"
                "使用全新优化器重新开始优化"
            )
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_selection_score = checkpoint.get(
            "best_selection_score",
            checkpoint.get("best_mae", float("inf")),
        )
        best_soft_selection_score = checkpoint.get(
            "best_soft_selection_score",
            best_selection_score,
        )

        old_best = os.path.join(
            args.save_dir, "best_soft.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_soft_pre_resume.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    f"旧 best_soft.pt 已备份到 {backup}"
                )

        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True

    for epoch in range(start_epoch, args.epochs):
        hard_route = False
        router_grad = epoch >= args.router_warmup_epochs
        temperature = temperature_for_epoch(
            epoch,
            hard_route,
            None,
            args,
        )

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            logging.info(
                "解冻 YOLO Backbone+Neck（保留优化器状态，不重建）"
            )

        model.train()
        total_loss = 0.0
        num_batches = 0
        matched_gate_sum = torch.zeros(
            3, device=device
        )
        matched_gate_top1_sum = torch.zeros(
            3, device=device
        )
        matched_gate_points = 0
        matched_gate_entropy_sum = torch.zeros(
            (), device=device
        )
        loss_sums = {
            name: torch.zeros((), device=device)
            for name in ("cls", "point", "count")
        }

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            images = batch["img"].to(device)
            gt_points = [
                p.to(device) for p in batch["points"]
            ]

            predictions = model(
                images,
                temperature=temperature,
                hard_route=hard_route,
                router_grad=router_grad,
            )

            loss, loss_items = criterion(
                predictions,
                gt_points,
                image_size=images.shape[-2:],
            )

            matched_gate_sum += loss_items[
                "matched_gate_hist"
            ].to(device)
            matched_gate_top1_sum += loss_items[
                "matched_gate_top1_hist"
            ].to(device)
            matched_gate_points += int(
                loss_items["matched_gate_count"].item()
            )
            matched_gate_entropy_sum += loss_items[
                "matched_gate_entropy"
            ].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )
            optimizer.step()

            total_loss += loss.item()
            for loss_name in loss_sums:
                loss_sums[loss_name] += loss_items[
                    loss_name
                ].detach()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_items = {
            loss_name: (
                loss_sums[loss_name]
                / max(num_batches, 1)
            ).item()
            for loss_name in loss_sums
        }

        (
            soft_mae,
            hard_mae,
            hard_usage,
            route_confusion,
            matched_points,
        ) = evaluate_count_mae(
            model,
            val_loader,
            device,
            args.match_top_k,
            soft_temperature=temperature,
            knn_k=criterion.knn_k,
            scale_centers=criterion.scale_centers,
            scale_sigma_octaves=criterion.scale_sigma_octaves,
            diagnose_scale_routing=(
                args.diagnose_scale_routing
            ),
        )

        normalizer = max(val_mean_gt_count, 1e-12)
        soft_weighted_norm_mae = soft_mae / normalizer
        hard_weighted_norm_mae = hard_mae / normalizer
        hard_soft_gap = (
            hard_weighted_norm_mae
            - soft_weighted_norm_mae
        )
        hard_soft_ratio = (
            hard_weighted_norm_mae
            / max(soft_weighted_norm_mae, 1e-12)
        )

        gate_mean = (
            matched_gate_sum
            / max(matched_gate_points, 1)
        )
        top1_usage = (
            matched_gate_top1_sum
            / max(matched_gate_points, 1)
        )
        router_entropy = (
            matched_gate_entropy_sum
            / max(matched_gate_points, 1)
        ).item()
        val_usage_pct = (
            hard_usage.float()
            / max(int(hard_usage.sum()), 1)
            * 100
        )

        logging.info(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} "
            f"cls={avg_loss_items['cls']:.4f} "
            f"point={avg_loss_items['point']:.4f} "
            f"count={avg_loss_items['count']:.4f} "
            f"T={temperature:.2f} "
            f"router_warmup={not router_grad} "
            f"soft_MAE={soft_mae:.3f} "
            f"hard_MAE={hard_mae:.3f} "
            f"soft_weighted_norm_mae="
            f"{soft_weighted_norm_mae:.6f} "
            f"hard_weighted_norm_mae="
            f"{hard_weighted_norm_mae:.6f} "
            f"hard_soft_gap={hard_soft_gap:.6f} "
            f"hard_soft_ratio={hard_soft_ratio:.4f}"
        )
        logging.info(
            "  matched gate mean=E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| matched top1=E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| val hard top1=E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| gate entropy=%.4f",
            gate_mean[0] * 100,
            gate_mean[1] * 100,
            gate_mean[2] * 100,
            top1_usage[0] * 100,
            top1_usage[1] * 100,
            top1_usage[2] * 100,
            val_usage_pct[0],
            val_usage_pct[1],
            val_usage_pct[2],
            router_entropy,
        )

        if args.diagnose_scale_routing and matched_points > 0:
            confusion_pct = (
                100.0
                * route_confusion.float()
                / route_confusion.sum(
                    dim=1, keepdim=True
                ).clamp_min(1)
            )
            row_counts = route_confusion.sum(dim=1)
            logging.info(
                "  scale-routing confusion "
                "(diagnostic only; GT class -> predicted expert):"
            )
            for expert_index in range(3):
                logging.info(
                    f"    GT E{expert_index} -> "
                    f"E0:{confusion_pct[expert_index, 0]:5.1f}% "
                    f"E1:{confusion_pct[expert_index, 1]:5.1f}% "
                    f"E2:{confusion_pct[expert_index, 2]:5.1f}% "
                    f"(n={int(row_counts[expert_index])})"
                )

        val_score_for_best = soft_weighted_norm_mae
        improved = val_score_for_best < best_soft_selection_score
        if improved:
            best_selection_score = val_score_for_best
            best_soft_selection_score = val_score_for_best
            best_path = os.path.join(
                args.save_dir, "best_soft.pt"
            )
        else:
            best_selection_score = best_soft_selection_score
            best_path = None

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_selection_score,
            "best_selection_score": best_selection_score,
            "best_soft_selection_score": (
                best_soft_selection_score
            ),
            "selection_metric": (
                "soft weighted normalized MAE"
            ),
            "hard_started": False,
            "hard_route": False,
            "soft_MAE": soft_mae,
            "hard_MAE": hard_mae,
            "soft_weighted_norm_mae": (
                soft_weighted_norm_mae
            ),
            "hard_weighted_norm_mae": (
                hard_weighted_norm_mae
            ),
            "hard_soft_gap": hard_soft_gap,
            "hard_soft_ratio": hard_soft_ratio,
            "gate_mean": gate_mean.detach().cpu(),
            "top1_usage": top1_usage.detach().cpu(),
            "gate_entropy": router_entropy,
            "val_hard_expert_usage": hard_usage.detach().cpu(),
            "router_training_mode": "task_only",
            "router_warmup_epochs": (
                args.router_warmup_epochs
            ),
            "diagnose_scale_routing": (
                args.diagnose_scale_routing
            ),
            "router_confusion": (
                route_confusion.detach().cpu()
                if args.diagnose_scale_routing
                else None
            ),
            "args": vars(args),
            "config": build_checkpoint_config(
                args,
                criterion,
                temperature=temperature,
            ),
        }
        if improved:
            torch.save(checkpoint_data, best_path)
            logging.info(
                f"  -> 新的最佳 soft weighted normalized MAE: "
                f"{best_selection_score:.6f} ({best_path})"
            )

        torch.save(
            checkpoint_data,
            os.path.join(args.save_dir, "last.pt"),
        )

    logging.info("训练结束。")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "训练 YOLO11 + Task-Only MoE Point Head；"
            "全程 soft routing"
        )
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11m.pt",
        help="YOLO11 预训练权重"
    )
    parser.add_argument(
        "--data-root", type=str,
        default="datasets/shanghaitech_AB",
        help="数据集根目录（含 images/ 与 points/）"
    )
    parser.add_argument(
        "--crop-size", type=int, default=640,
        help="训练/验证裁剪尺寸"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8
    )
    parser.add_argument(
        "--epochs", type=int, default=100
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=256
    )
    parser.add_argument(
        "--num-references", type=int, default=4,
        help="每个网格参考点数 K（1/4/9）"
    )
    parser.add_argument(
        "--scale-centers",
        type=str,
        default="10,20,40",
        help="可选尺度路由诊断的中心（像素）"
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--head-lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4
    )
    parser.add_argument(
        "--init-temperature", type=float, default=2.0,
        help="soft 阶段起始温度"
    )
    parser.add_argument(
        "--min-temperature", type=float, default=0.5,
        help="保留的 hard 兼容路径温度下限"
    )
    parser.add_argument(
        "--phase1-temp", type=float, default=1.3,
        help="soft baseline 第一段结束温度（默认 epoch 15 到达）"
    )
    parser.add_argument(
        "--soft-temp-floor", type=float, default=1.0,
        help="soft 阶段温度下限"
    )
    parser.add_argument(
        "--temp-floor-epoch", type=int, default=30,
        help="soft 温度降到 soft-temp-floor 的 epoch"
    )
    parser.add_argument(
        "--hard-temp-epochs", type=int, default=20,
        help="保留的 hard 兼容路径温度衰减 epoch 数"
    )
    parser.add_argument(
        "--router-warmup-epochs", type=int, default=3,
        help="前 N 个 epoch 使用精确均匀 gate；默认 3"
    )
    parser.add_argument(
        "--force-hard-epoch", type=int, default=None,
        help="已废弃并忽略；Unsup-v0 始终全程 soft"
    )
    parser.add_argument(
        "--diagnose-scale-routing",
        action="store_true",
        help="可选：输出 diagnostic-only 尺度路由混淆矩阵"
    )
    parser.add_argument(
        "--match-top-k", type=int, default=2000,
        help="匈牙利匹配的候选点 top-K（K=max(K, n_gt)）"
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=3,
        help="前 N 个 epoch 冻结 YOLO 部分（0 表示不冻结）"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=10.0
    )
    parser.add_argument(
        "--workers", type=int, default=4
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="权重保存目录（默认 runs/moe_point_<时间戳>；"
        "--resume 时沿用原 run 目录）"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 task_only checkpoint 恢复训练"
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
