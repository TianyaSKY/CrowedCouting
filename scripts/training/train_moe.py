import argparse
import logging
import math
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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.yolo11_moe_point import YOLO11MoEPoint
from models.point_moe_loss import PointMoELoss
from scripts.data.point_dataset import (
    PointDataset,
    point_collate_fn,
)
from scripts.visualization.validation_visualizer import (
    log_validation_images,
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
    router_active: bool,
) -> dict[str, object]:
    """Build the canonical D2 task-only Drop-1/Soft Top-2 settings."""
    temperature_schedule = {
        name: getattr(args, name)
        for name in (
            "init_temperature",
            "phase1_temp",
            "soft_temp_floor",
            "temp_floor_epoch",
        )
    }
    return {
        "crop_size": int(args.crop_size),
        "num_references": int(args.num_references),
        "temperature_schedule": temperature_schedule,
        "temperature": float(temperature),
        "router_training_mode": "task_only_drop1_soft_top2",
        "expert_dropout": "candidate_drop1",
        "active_experts": 2,
        "router_start_epoch": int(
            args.router_warmup_epochs
        ),
        "router_active": bool(router_active),
        "selection_metric": (
            "top2 weighted normalized MAE"
        ),
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
    ``gates`` is the actual forward assignment (random Drop-1 during
    training, deterministic full3/Top-2/Top-1 during evaluation).
    Entropy and margin use the complete Router soft probabilities so
    they remain informative even when the actual gate is masked.
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
    temperature: float = 1.0,
    knn_k: int = 1,
    scale_centers: tuple[float, float, float] = (
        10.0,
        20.0,
        40.0,
    ),
    scale_sigma_octaves: float = 0.6,
    diagnose_scale_routing: bool = False,
    criterion: PointMoELoss | None = None,
    max_visual_samples: int = 0,
):
    """Validate full3-soft, deterministic Top-2, and diagnostic Top-1.

    When ``criterion`` is supplied, Top-2 validation loss is computed from
    the already-produced deterministic Top-2 predictions. When
    ``max_visual_samples`` is positive, the same Top-2 predictions are
    detached and returned for TensorBoard rendering; no extra forward pass
    is performed.
    """
    if max_visual_samples < 0:
        raise ValueError("max_visual_samples must not be negative")

    was_training = model.training
    model.eval()

    mode_specs = {
        "full3": "full3_soft",
        "top2": "top2",
        "top1": "top1",
    }
    total_abs_error = {
        mode: 0.0 for mode in mode_specs
    }
    total_squared_error = {
        mode: 0.0 for mode in mode_specs
    }
    total_bias = {
        mode: 0.0 for mode in mode_specs
    }
    total_images = 0
    num_batches = 0
    top2_loss_sums = (
        {
            name: torch.zeros((), device=device)
            for name in ("total", "cls", "point", "count")
        }
        if criterion is not None
        else None
    )
    full3_top1_usage = torch.zeros(
        3, dtype=torch.int64, device=device
    )
    route_confusion = torch.zeros(
        3, 3, dtype=torch.int64, device=device
    )
    matched_points = 0
    full3_gate_entropy_sum = torch.zeros(
        (), device=device
    )
    full3_gate_margin_sum = torch.zeros(
        (), device=device
    )
    full3_gate_points = 0
    validation_samples: list[dict[str, object]] = []

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

            for mode, routing_mode in mode_specs.items():
                predictions = model(
                    images,
                    temperature=temperature,
                    routing_mode=routing_mode,
                )
                scores = predictions["logits"].sigmoid()
                pred_counts = scores.sum(dim=1)

                for image_index, gt_count in enumerate(gt_counts):
                    error = (
                        float(pred_counts[image_index].item())
                        - gt_count
                    )
                    total_abs_error[mode] += abs(error)
                    total_squared_error[mode] += error * error
                    total_bias[mode] += error

                if mode == "top2":
                    if criterion is not None:
                        top2_loss, loss_items = criterion(
                            predictions,
                            gt_points,
                            image_size=images.shape[-2:],
                        )
                        top2_loss_sums["total"] += (
                            top2_loss.detach()
                        )
                        for loss_name in ("cls", "point", "count"):
                            top2_loss_sums[loss_name] += (
                                loss_items[loss_name].detach()
                            )

                    if (
                        len(validation_samples)
                        < max_visual_samples
                    ):
                        image_paths = batch.get(
                            "image_paths", []
                        )
                        for image_index in range(
                            len(gt_counts)
                        ):
                            if (
                                len(validation_samples)
                                >= max_visual_samples
                            ):
                                break
                            image_path = None
                            if isinstance(image_paths, list):
                                image_path = image_paths[
                                    image_index
                                ]
                            validation_samples.append(
                                {
                                    "image": images[
                                        image_index
                                    ].detach().cpu(),
                                    "gt_points": gt_points[
                                        image_index
                                    ].detach().cpu(),
                                    "predictions": {
                                        name: predictions[name][
                                            image_index
                                        ].detach().cpu()
                                        for name in (
                                            "logits",
                                            "points",
                                            "gates",
                                        )
                                    },
                                    "image_path": image_path,
                                }
                            )

                if mode != "full3":
                    continue

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
                    scale_sigma_octaves=scale_sigma_octaves,
                    diagnose_scale_routing=(
                        diagnose_scale_routing
                    ),
                )
                full3_top1_usage += batch_usage
                full3_gate_entropy_sum += batch_entropy
                full3_gate_margin_sum += batch_margin
                full3_gate_points += batch_points
                matched_points += batch_matched
                if diagnose_scale_routing:
                    route_confusion += batch_confusion

            total_images += len(gt_counts)
            num_batches += 1

    if was_training:
        model.train()

    result = {
        "mae": {
            mode: total_abs_error[mode]
            / max(total_images, 1)
            for mode in mode_specs
        },
        "rmse": {
            mode: (
                total_squared_error[mode]
                / max(total_images, 1)
            ) ** 0.5
            for mode in mode_specs
        },
        "bias": {
            mode: total_bias[mode]
            / max(total_images, 1)
            for mode in mode_specs
        },
        "full3_top1_usage": full3_top1_usage,
        "route_confusion": route_confusion,
        "matched_points": matched_points,
        "full3_gate_entropy_sum": full3_gate_entropy_sum,
        "full3_gate_margin_sum": full3_gate_margin_sum,
        "full3_gate_points": full3_gate_points,
        "validation_samples": validation_samples,
    }
    if top2_loss_sums is None:
        result["top2_loss"] = None
    else:
        result["top2_loss"] = {
            name: (value / max(num_batches, 1)).item()
            for name, value in top2_loss_sums.items()
        }
    return result


def evaluate_expert_only_mae(
    model,
    val_loader,
    device,
    temperature: float = 1.0,
):
    """Evaluate each expert alone for periodic starvation diagnostics."""
    was_training = model.training
    model.eval()
    total_abs_error = [0.0, 0.0, 0.0]
    total_images = 0

    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Expert-only 验证中",
            leave=False,
        ):
            images = batch["img"].to(device)
            gt_counts = [
                points.shape[0] for points in batch["points"]
            ]
            for expert_index in range(3):
                predictions = model(
                    images,
                    temperature=temperature,
                    routing_mode="expert_only",
                    expert_index=expert_index,
                )
                pred_counts = (
                    predictions["logits"].sigmoid().sum(dim=1)
                )
                for image_index, gt_count in enumerate(
                    gt_counts
                ):
                    total_abs_error[expert_index] += abs(
                        float(pred_counts[image_index])
                        - gt_count
                    )
            total_images += len(gt_counts)

    if was_training:
        model.train()
    return {
        f"E{expert_index}": total_abs_error[expert_index]
        / max(total_images, 1)
        for expert_index in range(3)
    }


def temperature_for_epoch(
    epoch: int,
    router_active: bool,
    router_start_epoch: int,
    args,
) -> float:
    """Return T_router, restarting at 2.0 when Router becomes trainable."""
    if not router_active:
        return float(args.init_temperature)

    router_epoch = max(epoch - router_start_epoch, 0)
    phase1_epoch = max(
        int(args.temp_floor_epoch) // 2,
        1,
    )
    if router_epoch <= phase1_epoch:
        progress = router_epoch / phase1_epoch
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
            router_epoch - phase1_epoch,
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


def validation_image_options(args) -> tuple[int, int, float]:
    interval = int(getattr(args, "val_image_interval", 1))
    count = int(getattr(args, "val_image_count", 4))
    conf_threshold = float(
        getattr(args, "val_image_conf", 0.5)
    )
    if interval < 0:
        raise ValueError("--val-image-interval 不能为负数")
    if count < 0:
        raise ValueError("--val-image-count 不能为负数")
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("--val-image-conf 必须在 0 到 1 之间")
    return interval, count, conf_threshold


def dataset_tag_from_root(data_root: str) -> str:
    tag = os.path.basename(os.path.normpath(data_root))
    return tag or "dataset"


def log_system_metrics(
    writer: SummaryWriter,
    optimizer,
    epoch: int,
    epoch_seconds: float,
    device: str,
) -> None:
    writer.add_scalar(
        "system/epoch_seconds",
        epoch_seconds,
        epoch,
    )
    if str(device).startswith("cuda") and torch.cuda.is_available():
        allocated_mb = (
            torch.cuda.memory_allocated() / (1024 ** 2)
        )
        reserved_mb = (
            torch.cuda.memory_reserved() / (1024 ** 2)
        )
    else:
        allocated_mb = 0.0
        reserved_mb = 0.0
    writer.add_scalar(
        "system/gpu_memory_allocated_mb",
        allocated_mb,
        epoch,
    )
    writer.add_scalar(
        "system/gpu_memory_reserved_mb",
        reserved_mb,
        epoch,
    )
    if optimizer.param_groups:
        writer.add_scalar(
            "train/lr_head",
            float(optimizer.param_groups[0]["lr"]),
            epoch,
        )
    if len(optimizer.param_groups) > 1:
        writer.add_scalar(
            "train/lr_backbone",
            float(optimizer.param_groups[1]["lr"]),
            epoch,
        )


def train_moe(args):
    if args.router_warmup_epochs < 0:
        raise ValueError("--router-warmup-epochs 不能为负数")

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

    # TensorBoard
    tb_dir = os.path.join(args.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info("TensorBoard 日志目录: %s", tb_dir)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"使用设备: {device}")
    validate_cuda_device(device)

    # 覆盖保护：从头训练进入已有 save-dir 时备份旧 best_top2。
    if not args.resume:
        old_best = os.path.join(
            args.save_dir, "best_top2.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_top2_prev.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    f"旧 best_top2.pt 已备份到 {backup}"
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
    val_image_interval, val_image_count, val_image_conf = (
        validation_image_options(args)
    )
    dataset_tag = dataset_tag_from_root(args.data_root)


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
    for param in model.point_head.router.parameters():
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
            != "task_only_drop1_soft_top2"
        ):
            raise ValueError(
                "D2 只能从 router_training_mode="
                "'task_only_drop1_soft_top2' checkpoint 恢复；"
                "禁止从旧 H0 或 soft-only checkpoint resume"
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
            args.save_dir, "best_top2.pt"
        )
        if os.path.exists(old_best):
            backup = os.path.join(
                args.save_dir, "best_top2_pre_resume.pt"
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    f"旧 best_top2.pt 已备份到 {backup}"
                )


        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
        if start_epoch >= args.router_warmup_epochs:
            for param in model.point_head.router.parameters():
                param.requires_grad = True


    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.perf_counter()
        router_warmup = (
            epoch < args.router_warmup_epochs
        )
        router_active = not router_warmup
        router_grad = router_active
        routing_mode = "train_drop1"
        temperature = temperature_for_epoch(
            epoch,
            router_active,
            args.router_warmup_epochs,
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
        if epoch == args.router_warmup_epochs:
            for param in model.point_head.router.parameters():
                param.requires_grad = True
            logging.info(
                "启用 Router task-only Drop-1 soft Top-2 "
                "（温度从 T_router=2.0 重新开始）"
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
        matched_gate_points = 0
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
                routing_mode=routing_mode,
                router_grad=router_grad,
            )
            batch_usage, _, _, _ = routing_statistics(
                predictions
            )
            train_sampled_usage += batch_usage

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

        collect_visuals = (
            val_image_interval > 0
            and (epoch + 1) % val_image_interval == 0
        )
        validation = evaluate_count_mae(
            model,
            val_loader,
            device,
            args.match_top_k,
            temperature=temperature,
            knn_k=criterion.knn_k,
            scale_centers=criterion.scale_centers,
            scale_sigma_octaves=criterion.scale_sigma_octaves,
            diagnose_scale_routing=(
                args.diagnose_scale_routing
            ),
            criterion=criterion,
            max_visual_samples=(
                val_image_count if collect_visuals else 0
            ),
        )

        full3_mae = validation["mae"]["full3"]
        top2_mae = validation["mae"]["top2"]
        top1_mae = validation["mae"]["top1"]
        full3_rmse = validation["rmse"]["full3"]
        top2_rmse = validation["rmse"]["top2"]
        top1_rmse = validation["rmse"]["top1"]
        full3_bias = validation["bias"]["full3"]
        top2_bias = validation["bias"]["top2"]
        top1_bias = validation["bias"]["top1"]
        top2_loss = validation["top2_loss"]
        normalizer = max(val_mean_gt_count, 1e-12)
        full3_norm_mae = full3_mae / normalizer
        top2_norm_mae = top2_mae / normalizer
        top1_norm_mae = top1_mae / normalizer
        full3_weighted_norm_mae = full3_norm_mae
        top2_weighted_norm_mae = top2_norm_mae
        top1_weighted_norm_mae = top1_norm_mae
        top2_full3_gap = (
            top2_weighted_norm_mae
            - full3_weighted_norm_mae
        )
        top1_top2_gap = (
            top1_weighted_norm_mae
            - top2_weighted_norm_mae
        )

        matched_probability_mean = (
            matched_probability_sum
            / max(matched_gate_points, 1)
        )
        matched_top1_usage = (
            matched_top1_sum
            / max(matched_gate_points, 1)
        )
        val_gate_entropy = (
            validation["full3_gate_entropy_sum"]
            / max(validation["full3_gate_points"], 1)
        ).item()
        val_gate_margin = (
            validation["full3_gate_margin_sum"]
            / max(validation["full3_gate_points"], 1)
        ).item()
        full3_top1_usage = validation[
            "full3_top1_usage"
        ]
        full3_top1_usage_pct = (
            full3_top1_usage.float()
            / max(int(full3_top1_usage.sum()), 1)
            * 100
        )
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
        expert_only_mae = None
        if (
            args.expert_only_eval_interval > 0
            and (epoch + 1)
            % args.expert_only_eval_interval
            == 0
        ):
            expert_only_mae = evaluate_expert_only_mae(
                model,
                val_loader,
                device,
                temperature=temperature,
            )

        logging.info(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} "
            f"cls={avg_loss_items['cls']:.4f} "
            f"point={avg_loss_items['point']:.4f} "
            f"count={avg_loss_items['count']:.4f} "
            f"T_router={temperature:.2f} "
            f"routing={routing_mode} "
            f"router_active={router_active} "
            f"MAE_full3={full3_mae:.3f}/"
            f"{full3_weighted_norm_mae:.6f} "
            f"MAE_top2={top2_mae:.3f}/"
            f"{top2_weighted_norm_mae:.6f} "
            f"MAE_top1={top1_weighted_norm_mae:.6f} "
            f"top2_full3_gap={top2_full3_gap:.6f} "
            f"top1_top2_gap={top1_top2_gap:.6f}"
        )
        logging.info(
            "  full3 matched probability mean="
            "E0:%.1f%% E1:%.1f%% E2:%.1f%% "
            "| full3 entropy=%.4f margin=%.4f "
            "| full3 deterministic Top-1="
            "E0:%.1f%% E1:%.1f%% E2:%.1f%%",
            matched_probability_mean[0] * 100,
            matched_probability_mean[1] * 100,
            matched_probability_mean[2] * 100,
            val_gate_entropy,
            val_gate_margin,
            full3_top1_usage_pct[0],
            full3_top1_usage_pct[1],
            full3_top1_usage_pct[2],
        )
        logging.info(
            "  train sampled usage=%s "
            "| matched train gate Top-1=%s",
            train_usage_string,
            matched_top1_usage_string,
        )
        if expert_only_mae is not None:
            logging.info(
                "  expert-only MAE: E0=%.3f E1=%.3f E2=%.3f",
                expert_only_mae["E0"],
                expert_only_mae["E1"],
                expert_only_mae["E2"],
            )

        route_confusion = validation["route_confusion"]
        matched_points = validation["matched_points"]
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

        # ---- TensorBoard 写入 ----
        # Preserve the historical tags for existing event files.
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/cls", avg_loss_items["cls"], epoch)
        writer.add_scalar("loss/point", avg_loss_items["point"], epoch)
        writer.add_scalar("loss/count", avg_loss_items["count"], epoch)

        writer.add_scalar("train/loss_total", avg_loss, epoch)
        writer.add_scalar(
            "train/loss_cls", avg_loss_items["cls"], epoch
        )
        writer.add_scalar(
            "train/loss_point", avg_loss_items["point"], epoch
        )
        writer.add_scalar(
            "train/loss_count", avg_loss_items["count"], epoch
        )

        if top2_loss is not None:
            for loss_name in ("total", "cls", "point", "count"):
                writer.add_scalar(
                    f"val/loss_top2_{loss_name}",
                    top2_loss[loss_name],
                    epoch,
                )

        writer.add_scalar("mae/full3_raw", full3_mae, epoch)
        writer.add_scalar("mae/top2_raw", top2_mae, epoch)
        writer.add_scalar(
            "mae/full3_weighted_norm",
            full3_weighted_norm_mae,
            epoch,
        )
        writer.add_scalar(
            "mae/top2_weighted_norm",
            top2_weighted_norm_mae,
            epoch,
        )
        writer.add_scalar(
            "mae/top1_weighted_norm",
            top1_weighted_norm_mae,
            epoch,
        )
        writer.add_scalar(
            "val/mae_full3_raw", full3_mae, epoch
        )
        writer.add_scalar(
            "val/mae_top2_raw", top2_mae, epoch
        )
        writer.add_scalar(
            "val/mae_top1_raw", top1_mae, epoch
        )
        writer.add_scalar(
            "val/mae_full3_weighted_norm",
            full3_weighted_norm_mae,
            epoch,
        )
        writer.add_scalar(
            "val/mae_top2_weighted_norm",
            top2_weighted_norm_mae,
            epoch,
        )
        writer.add_scalar(
            "val/mae_top1_weighted_norm",
            top1_weighted_norm_mae,
            epoch,
        )
        for mode, rmse, bias in (
            ("full3", full3_rmse, full3_bias),
            ("top2", top2_rmse, top2_bias),
            ("top1", top1_rmse, top1_bias),
        ):
            writer.add_scalar(
                f"val/rmse_{mode}", rmse, epoch
            )
            writer.add_scalar(
                f"val/count_bias_{mode}", bias, epoch
            )
        writer.add_scalar(
            "val/rmse_top2", top2_rmse, epoch
        )
        writer.add_scalar(
            "val/count_bias", top2_bias, epoch
        )
        writer.add_scalar(
            "schedule/router_temperature", temperature, epoch
        )

        writer.add_scalar("routing/full3_entropy", val_gate_entropy, epoch)
        writer.add_scalar("routing/full3_margin", val_gate_margin, epoch)
        writer.add_scalar(
            "val/routing_entropy", val_gate_entropy, epoch
        )
        writer.add_scalar(
            "val/routing_margin", val_gate_margin, epoch
        )
        for expert_index in range(3):
            writer.add_scalar(
                f"routing/full3_top1_usage_E{expert_index}_pct",
                float(full3_top1_usage_pct[expert_index]),
                epoch,
            )
            writer.add_scalar(
                f"val/expert_usage_E{expert_index}",
                float(full3_top1_usage_pct[expert_index]) / 100.0,
                epoch,
            )
        if expert_only_mae is not None:
            for expert_index in range(3):
                writer.add_scalar(
                    f"mae/expert_only_E{expert_index}",
                    expert_only_mae[f"E{expert_index}"],
                    epoch,
                )

        if collect_visuals and validation["validation_samples"]:
            log_validation_images(
                writer,
                f"val_images/{dataset_tag}",
                validation["validation_samples"],
                epoch,
                conf_threshold=val_image_conf,
            )

        val_score_for_best = top2_weighted_norm_mae
        improved = (
            not router_warmup
            and val_score_for_best < best_selection_score
        )
        if improved:
            best_selection_score = val_score_for_best
            best_path = os.path.join(
                args.save_dir, "best_top2.pt"
            )
        else:
            best_path = None
        if math.isfinite(best_selection_score):
            writer.add_scalar(
                "val/best_score",
                best_selection_score,
                epoch,
            )


        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_selection_score,
            "best_selection_score": best_selection_score,
            "selection_metric": (
                "top2 weighted normalized MAE"
            ),
            "MAE_full3": full3_mae,
            "MAE_top2": top2_mae,
            "full3_raw_mae": full3_mae,
            "top2_raw_mae": top2_mae,
            "full3_norm_mae": full3_norm_mae,
            "top2_norm_mae": top2_norm_mae,
            "top1_norm_mae": top1_norm_mae,
            "full3_weighted_norm_mae": (
                full3_weighted_norm_mae
            ),
            "top2_weighted_norm_mae": (
                top2_weighted_norm_mae
            ),
            "top1_weighted_norm_mae": (
                top1_weighted_norm_mae
            ),
            "top2_full3_gap": top2_full3_gap,
            "top1_top2_gap": top1_top2_gap,
            "top2_loss": top2_loss,
            "full3_rmse": full3_rmse,
            "top2_rmse": top2_rmse,
            "top1_rmse": top1_rmse,
            "full3_count_bias": full3_bias,
            "top2_count_bias": top2_bias,
            "top1_count_bias": top1_bias,
            "full3_matched_probability_mean": (
                matched_probability_mean.detach().cpu()
            ),
            "matched_top1_usage": (
                matched_top1_usage_checkpoint
            ),
            "full3_top1_usage": (
                full3_top1_usage.detach().cpu()
            ),
            "full3_gate_entropy": val_gate_entropy,
            "full3_gate_margin": val_gate_margin,
            "train_sampled_usage": (
                train_sampled_usage_checkpoint
            ),
            "expert_only_mae": expert_only_mae,
            "router_training_mode": (
                "task_only_drop1_soft_top2"
            ),
            "expert_dropout": "candidate_drop1",
            "active_experts": 2,
            "router_start_epoch": args.router_warmup_epochs,
            "router_active": bool(router_active),
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
                router_active=router_active,
            ),
        }
        if improved:
            torch.save(checkpoint_data, best_path)
            logging.info(
                f"  -> 新的最佳 top2 weighted normalized MAE: "
                f"{best_selection_score:.6f} ({best_path})"
            )

        torch.save(
            checkpoint_data,
            os.path.join(args.save_dir, "last.pt"),
        )
        epoch_seconds = (
            time.perf_counter() - epoch_started_at
        )
        log_system_metrics(
            writer,
            optimizer,
            epoch,
            epoch_seconds,
            device,
        )
        writer.flush()

    writer.close()
    logging.info("训练结束。")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "训练 YOLO11 + D2 Task-Only MoE Point Head；"
            "candidate Drop-1 + Soft Top-2"
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
        help="Router 启用时 T_router 的起始温度"
    )
    parser.add_argument(
        "--phase1-temp", type=float, default=1.5,
        help="T_router 第一段结束温度（默认 Router epoch 15 到达）"
    )
    parser.add_argument(
        "--soft-temp-floor", type=float, default=1.0,
        help="T_router 下限"
    )
    parser.add_argument(
        "--temp-floor-epoch", type=int, default=30,
        help="Router epoch 到达 soft-temp-floor 的时间点"
    )
    parser.add_argument(
        "--router-warmup-epochs", type=int, default=6,
        help="前 N 个 epoch Router 冻结且使用随机 candidate Drop-1"
    )
    parser.add_argument(
        "--expert-only-eval-interval", type=int, default=5,
        help="每 N 个 epoch 运行一次 E0/E1/E2-only 诊断；0 表示关闭"
    )
    parser.add_argument(
        "--val-image-interval",
        type=int,
        default=1,
        help="每 N 个 epoch 写入固定验证图；0 表示关闭",
    )
    parser.add_argument(
        "--val-image-count",
        type=int,
        default=4,
        help="每次写入的验证图数量",
    )
    parser.add_argument(
        "--val-image-conf",
        type=float,
        default=0.5,
        help="验证图可视化的可见点阈值；Metric count 仍使用 sigmoid 求和",
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
        help="仅从 router_training_mode=task_only_drop1_soft_top2 checkpoint 恢复"
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
