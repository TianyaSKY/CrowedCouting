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
    hard_route: bool,
) -> dict[str, object]:
    """Build the canonical Hard-Only Task-Driven MoE settings."""
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
        "hard_started": bool(hard_route),
        "hard_route": bool(hard_route),
        "training_hard_route": bool(hard_route),
        "router_training_mode": "task_only_gumbel_hard",
        "router_warmup_epochs": int(
            args.router_warmup_epochs
        ),
        "route_supervision": False,
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
    diagnose_scale_routing: bool = False,
) -> tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Match validation positives and collect routing diagnostics.

    Matching uses the same top-K Hungarian cost as ``PointMoELoss``.
    Deterministic hard usage uses ``gates``; entropy and margin use
    ``route_probabilities`` at the caller-selected temperature. The
    optional scale confusion matrix remains diagnostic-only.
    """
    from models.point_moe_loss import PointMoELoss

    logits = predictions["logits"]
    points = predictions["points"]
    gates = predictions["gates"]
    route_probabilities = predictions.get(
        "route_probabilities"
    )
    if route_probabilities is None:
        route_probabilities = torch.softmax(
            predictions["route_logits"],
            dim=2,
        )
    route_probabilities = route_probabilities.permute(
        0, 3, 4, 1, 2
    ).reshape_as(gates)

    batch_size = logits.shape[0]
    image_height, image_width = image_size
    scale = points.new_tensor(
        [float(image_width), float(image_height)]
    )

    confusion = torch.zeros(
        3, 3, dtype=torch.int64, device=device
    )
    confusion_matched_points = 0
    matched_usage = torch.zeros(
        3, dtype=torch.int64, device=device
    )
    matched_entropy = points.new_zeros(())
    matched_margin = points.new_zeros(())
    matched_route_points = 0

    for batch_index in range(batch_size):
        gt = gt_points[batch_index].to(device)
        number_of_gt = gt.shape[0]
        if number_of_gt == 0:
            continue

        pred_logits = logits[batch_index]
        pred_points = points[batch_index]
        pred_gates = gates[batch_index]

        top_k = max(match_top_k, number_of_gt)
        if pred_logits.shape[0] > top_k:
            match_indices = pred_logits.topk(
                top_k
            ).indices
            match_logits = pred_logits[match_indices]
            match_points = pred_points[match_indices]
        else:
            match_indices = None
            match_logits = pred_logits
            match_points = pred_points

        if match_points.shape[0] < number_of_gt:
            raise RuntimeError("候选点数量小于真实点数量")

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

        matched_hard_gates = pred_gates[
            matched_full_indices
        ]
        matched_usage += matched_hard_gates.argmax(
            dim=-1
        ).bincount(minlength=3)
        matched_probabilities = route_probabilities[
            batch_index
        ][matched_full_indices]
        safe_probabilities = matched_probabilities.clamp_min(
            1e-8
        )
        matched_entropy += -(
            safe_probabilities * safe_probabilities.log()
        ).sum()
        margin_values = matched_probabilities.topk(
            k=2,
            dim=-1,
        ).values
        matched_margin += (
            margin_values[:, 0] - margin_values[:, 1]
        ).sum()
        matched_route_points += int(
            matched_full_indices.numel()
        )

        # Scale labels are optional and remain diagnostic-only.
        if (
            not diagnose_scale_routing
            or number_of_gt < knn_k + 1
        ):
            continue

        target_gate = PointMoELoss.scale_targets(
            gt,
            knn_k=knn_k,
            scale_centers=scale_centers,
            scale_sigma_octaves=scale_sigma_octaves,
        )
        target_class = target_gate.argmax(dim=1)
        pred_class = matched_hard_gates.argmax(dim=-1)

        confusion.index_add_(
            0,
            target_class,
            torch.nn.functional.one_hot(
                pred_class,
                num_classes=3,
            ).to(device),
        )
        confusion_matched_points += int(
            matched_full_indices.numel()
        )

    return (
        confusion,
        confusion_matched_points,
        matched_usage,
        matched_entropy.detach(),
        matched_margin.detach(),
        matched_route_points,
    )

def routing_statistics(
    predictions: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return sampled usage, entropy, margin, and selected-point count.

    ``gates`` is the actual forward assignment (Gumbel sample during
    training, deterministic argmax during evaluation). Entropy and margin
    use the Router's soft probabilities so they remain informative for a
    hard route.
    """
    gates = predictions["gates"]
    route_probabilities = predictions.get(
        "route_probabilities"
    )
    if route_probabilities is None:
        route_probabilities = torch.softmax(
            predictions["route_logits"],
            dim=2,
        )
    route_probabilities = route_probabilities.permute(
        0, 3, 4, 1, 2
    ).reshape_as(gates)

    if mask is None:
        selected_gates = gates.reshape(-1, gates.shape[-1])
        selected_probabilities = route_probabilities.reshape(
            -1, route_probabilities.shape[-1]
        )
    else:
        selected_gates = gates[mask]
        selected_probabilities = route_probabilities[mask]

    selected_count = int(selected_gates.shape[0])
    if selected_count == 0:
        return (
            torch.zeros(
                gates.shape[-1],
                dtype=torch.int64,
                device=gates.device,
            ),
            gates.new_zeros(()),
            gates.new_zeros(()),
            0,
        )

    sampled_usage = selected_gates.argmax(dim=-1).bincount(
        minlength=gates.shape[-1]
    )
    safe_probabilities = selected_probabilities.clamp_min(1e-8)
    entropy_sum = -(
        safe_probabilities * safe_probabilities.log()
    ).sum()
    margin_values = selected_probabilities.topk(
        k=2,
        dim=-1,
    ).values
    margin_sum = (
        margin_values[:, 0] - margin_values[:, 1]
    ).sum()
    return (
        sampled_usage,
        entropy_sum.detach(),
        margin_sum.detach(),
        selected_count,
    )


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
    """Validate soft diagnostics and deterministic hard primary metrics.

    Hard validation never samples Gumbel noise. Scale confusion remains
    diagnostic-only and does not affect training or checkpoint selection.
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
    hard_gate_entropy_sum = torch.zeros(
        (), device=device
    )
    hard_gate_margin_sum = torch.zeros(
        (), device=device
    )
    hard_gate_points = 0

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
                    (
                        batch_confusion,
                        batch_matched,
                        batch_usage,
                        batch_entropy,
                        batch_margin,
                        batch_points,
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
                        diagnose_scale_routing=(
                            diagnose_scale_routing
                        ),
                    )
                    hard_usage += batch_usage
                    hard_gate_entropy_sum += batch_entropy
                    hard_gate_margin_sum += batch_margin
                    hard_gate_points += batch_points
                    if diagnose_scale_routing:
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
        hard_gate_entropy_sum,
        hard_gate_margin_sum,
        hard_gate_points,
    )


def temperature_for_epoch(
    epoch: int,
    hard_route: bool,
    first_hard_epoch: int | None,
    args,
) -> float:
    """Return the shared 2.0 -> 1.3 -> 1.0 temperature schedule.

    Hard routing changes the forward assignment, not this schedule. The
    legacy parameters remain in the signature/config for checkpoint readers.
    """
    _ = hard_route, first_hard_epoch

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
            "--force-hard-epoch 已废弃；H0 在 Router warm-up 后自动 "
            "使用 Gumbel hard，参数被忽略"
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

    # 覆盖保护：从头训练进入已有 save-dir 时备份旧 best_hard。
    if not args.resume:
        old_best = os.path.join(
            args.save_dir, "best_hard.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_hard_prev.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    f"旧 best_hard.pt 已备份到 {backup}"
                )

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

    if args.resume and os.path.exists(args.resume):
        logging.info(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        saved_config = checkpoint.get("config", {})
        if (
            not isinstance(saved_config, dict)
            or saved_config.get("router_training_mode")
            != "task_only_gumbel_hard"
        ):
            raise ValueError(
                "H0 只能从 router_training_mode="
                "'task_only_gumbel_hard' checkpoint 恢复；"
                "禁止从 Unsup-v0 soft-only checkpoint resume"
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

        old_best = os.path.join(
            args.save_dir, "best_hard.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_hard_pre_resume.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    f"旧 best_hard.pt 已备份到 {backup}"
                )

        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True

    for epoch in range(start_epoch, args.epochs):
        router_warmup = (
            epoch < args.router_warmup_epochs
        )
        if router_warmup:
            hard_route = False
            router_grad = False
            routing_mode = "warmup_uniform"
        else:
            hard_route = True
            router_grad = True
            routing_mode = "train_gumbel_hard"
        temperature = temperature_for_epoch(
            epoch,
            hard_route,
            None,
            args,
        )
        if (
            args.freeze_epochs > 0
            and epoch == args.freeze_epochs
        ):
            for param in model.yolo.parameters():
                param.requires_grad = True
            logging.info(
                "解冻 YOLO Backbone+Neck（保留优化器状态，不重建）"
            )

        model.train()
        total_loss = 0.0
        num_batches = 0
        matched_probability_sum = torch.zeros(
            3, device=device
        )
        matched_top1_sum = torch.zeros(
            3, device=device
        )
        train_sampled_usage = torch.zeros(
            3, dtype=torch.int64, device=device
        )
        train_gate_entropy_sum = torch.zeros(
            (), device=device
        )
        train_gate_margin_sum = torch.zeros(
            (), device=device
        )
        train_gate_points = 0
        matched_gate_points = 0
        matched_gate_entropy_sum = torch.zeros(
            (), device=device
        )
        matched_gate_margin_sum = torch.zeros(
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
            (
                batch_usage,
                batch_entropy,
                batch_margin,
                batch_points,
            ) = routing_statistics(predictions)
            train_sampled_usage += batch_usage
            train_gate_entropy_sum += batch_entropy
            train_gate_margin_sum += batch_margin
            train_gate_points += batch_points

            loss, loss_items = criterion(
                predictions,
                gt_points,
                image_size=images.shape[-2:],
            )

            matched_probability_sum += loss_items[
                "matched_probability_sum"
            ].to(device)
            matched_top1_sum += loss_items[
                "matched_top1_hist"
            ].to(device)
            matched_gate_points += int(
                loss_items["matched_gate_count"].item()
            )
            matched_gate_entropy_sum += loss_items[
                "matched_gate_entropy"
            ].to(device)
            matched_gate_margin_sum += loss_items[
                "matched_gate_margin"
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
            val_gate_entropy_sum,
            val_gate_margin_sum,
            val_gate_points,
        ) = evaluate_count_mae(
            model,
            val_loader,
            device,
            args.match_top_k,
            soft_temperature=temperature,
            hard_temperature=temperature,
            knn_k=criterion.knn_k,
            scale_centers=criterion.scale_centers,
            scale_sigma_octaves=criterion.scale_sigma_octaves,
            diagnose_scale_routing=(
                args.diagnose_scale_routing
            ),
        )

        normalizer = max(val_mean_gt_count, 1e-12)
        soft_norm_mae = soft_mae / normalizer
        hard_norm_mae = hard_mae / normalizer
        soft_weighted_norm_mae = soft_norm_mae
        hard_weighted_norm_mae = hard_norm_mae
        hard_soft_gap = (
            hard_weighted_norm_mae
            - soft_weighted_norm_mae
        )
        hard_soft_ratio = (
            hard_weighted_norm_mae
            / max(soft_weighted_norm_mae, 1e-12)
        )

        matched_probability_mean = (
            matched_probability_sum
            / max(matched_gate_points, 1)
        )
        matched_top1_usage = (
            matched_top1_sum
            / max(matched_gate_points, 1)
        )
        matched_router_margin = (
            matched_gate_margin_sum
            / max(matched_gate_points, 1)
        ).item()
        router_entropy = (
            matched_gate_entropy_sum
            / max(matched_gate_points, 1)
        ).item()
        train_gate_entropy = (
            train_gate_entropy_sum
            / max(train_gate_points, 1)
        ).item()
        train_gate_margin = (
            train_gate_margin_sum
            / max(train_gate_points, 1)
        ).item()
        val_gate_entropy = (
            val_gate_entropy_sum
            / max(val_gate_points, 1)
        ).item()
        val_gate_margin = (
            val_gate_margin_sum
            / max(val_gate_points, 1)
        ).item()
        if router_warmup:
            train_usage_pct = None
            train_usage_string = "N/A (uniform warmup)"
            train_sampled_usage_checkpoint = None
            matched_top1_usage_string = (
                "N/A (uniform warmup)"
            )
            matched_top1_usage_checkpoint = None
        else:
            train_usage_pct = (
                train_sampled_usage.float()
                / max(int(train_sampled_usage.sum()), 1)
                * 100
            )
            train_usage_string = (
                f"E0:{train_usage_pct[0]:.1f}% "
                f"E1:{train_usage_pct[1]:.1f}% "
                f"E2:{train_usage_pct[2]:.1f}%"
            )
            train_sampled_usage_checkpoint = (
                train_sampled_usage.detach().cpu()
            )
            matched_top1_usage_string = (
                f"E0:{matched_top1_usage[0] * 100:.1f}% "
                f"E1:{matched_top1_usage[1] * 100:.1f}% "
                f"E2:{matched_top1_usage[2] * 100:.1f}%"
            )
            matched_top1_usage_checkpoint = (
                matched_top1_usage.detach().cpu()
            )
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
            f"routing={routing_mode} "
            f"router_warmup={router_warmup} "
            f"hard_raw_mae={hard_mae:.3f} "
            f"hard_norm_mae={hard_norm_mae:.6f} "
            f"hard_weighted_norm_mae="
            f"{hard_weighted_norm_mae:.6f} "
            f"soft_raw_mae={soft_mae:.3f} "
            f"soft_norm_mae={soft_norm_mae:.6f} "
            f"soft_weighted_norm_mae="
            f"{soft_weighted_norm_mae:.6f} "
            f"hard_soft_gap={hard_soft_gap:.6f} "
            f"hard_soft_ratio={hard_soft_ratio:.4f}"
        )
        logging.info(
            "  matched probability mean="
            "E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| matched sampled Top-1=%s "
            "| train sampled usage=%s "
            "| val matched deterministic usage="
            "E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| matched entropy=%.4f matched margin=%.4f "
            "| train entropy=%.4f val entropy=%.4f "
            "| train margin=%.4f val margin=%.4f",
            matched_probability_mean[0] * 100,
            matched_probability_mean[1] * 100,
            matched_probability_mean[2] * 100,
            matched_top1_usage_string,
            train_usage_string,
            val_usage_pct[0],
            val_usage_pct[1],
            val_usage_pct[2],
            router_entropy,
            matched_router_margin,
            train_gate_entropy,
            val_gate_entropy,
            train_gate_margin,
            val_gate_margin,
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

        val_score_for_best = hard_weighted_norm_mae
        improved = (
            not router_warmup
            and val_score_for_best < best_selection_score
        )
        if improved:
            best_selection_score = val_score_for_best
            best_path = os.path.join(
                args.save_dir, "best_hard.pt"
            )
        else:
            best_path = None

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_selection_score,
            "best_selection_score": best_selection_score,
            "best_hard_selection_score": (
                best_selection_score
            ),
            "selection_metric": (
                "hard weighted normalized MAE"
            ),
            "hard_started": bool(hard_route),
            "hard_route": bool(hard_route),
            "training_hard_route": bool(hard_route),
            "soft_MAE": soft_mae,
            "hard_MAE": hard_mae,
            "soft_raw_mae": soft_mae,
            "hard_raw_mae": hard_mae,
            "soft_norm_mae": soft_norm_mae,
            "hard_norm_mae": hard_norm_mae,
            "soft_weighted_norm_mae": (
                soft_weighted_norm_mae
            ),
            "hard_weighted_norm_mae": (
                hard_weighted_norm_mae
            ),
            "hard_soft_gap": hard_soft_gap,
            "hard_soft_ratio": hard_soft_ratio,
            "matched_probability_mean": (
                matched_probability_mean.detach().cpu()
            ),
            "matched_top1_usage": (
                matched_top1_usage_checkpoint
            ),
            "gate_entropy": val_gate_entropy,
            "gate_margin": val_gate_margin,
            "matched_gate_entropy": router_entropy,
            "matched_gate_margin": matched_router_margin,
            "train_gate_entropy": train_gate_entropy,
            "train_gate_margin": train_gate_margin,
            "val_gate_entropy": val_gate_entropy,
            "val_gate_margin": val_gate_margin,
            "train_sampled_usage": (
                train_sampled_usage_checkpoint
            ),
            "val_matched_deterministic_usage": (
                hard_usage.detach().cpu()
            ),
            "val_deterministic_usage": (
                hard_usage.detach().cpu()
            ),
            "val_hard_expert_usage": hard_usage.detach().cpu(),
            "router_training_mode": (
                "task_only_gumbel_hard"
            ),
            "router_warmup_epochs": (
                args.router_warmup_epochs
            ),
            "route_supervision": False,
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
                hard_route=hard_route,
            ),
        }
        if improved:
            torch.save(checkpoint_data, best_path)
            logging.info(
                f"  -> 新的最佳 hard weighted normalized MAE: "
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
            "训练 YOLO11 + Hard-Only Task-Driven MoE Point Head；"
            "Router warm-up 后使用 Gumbel-ST Top-1"
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
        help="Gumbel-ST backward surrogate 起始温度"
    )
    parser.add_argument(
        "--min-temperature", type=float, default=0.5,
        help="旧 checkpoint 兼容字段；当前共享 schedule 不使用"
    )
    parser.add_argument(
        "--phase1-temp", type=float, default=1.3,
        help="共享温度曲线第一段结束温度（默认 epoch 15 到达）"
    )
    parser.add_argument(
        "--soft-temp-floor", type=float, default=1.0,
        help="共享温度曲线下限"
    )
    parser.add_argument(
        "--temp-floor-epoch", type=int, default=30,
        help="soft 温度降到 soft-temp-floor 的 epoch"
    )
    parser.add_argument(
        "--hard-temp-epochs", type=int, default=20,
        help="旧 checkpoint 兼容字段；当前共享 schedule 不使用"
    )
    parser.add_argument(
        "--router-warmup-epochs", type=int, default=3,
        help="前 N 个 epoch 使用均匀 gate；之后使用 Gumbel-ST hard"
    )
    parser.add_argument(
        "--force-hard-epoch", type=int, default=None,
        help="已废弃并忽略；warm-up 后自动使用 Gumbel-ST hard"
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
        help="仅从 task_only_gumbel_hard checkpoint 恢复"
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
