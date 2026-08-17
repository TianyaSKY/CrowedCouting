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
    hard_started: bool,
    hard_route: bool,
    temperature: float,
) -> dict[str, object]:
    """Build the canonical train/eval settings stored in every checkpoint."""
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
        "hard_started": bool(hard_started),
        "hard_route": bool(hard_route),
        "router_grad_epoch": int(args.router_grad_epoch),
        "scale_centers": [
            float(center) for center in criterion.scale_centers
        ],
        "scale_sigma_octaves": float(criterion.scale_sigma_octaves),
        "expert_uniform_floor": float(
            getattr(args, "expert_uniform_floor", 0.0)
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
    """单个 batch 的路由混淆矩阵：GT 尺度目标类(行) x 预测专家(列)。

    匹配口径与 PointMoELoss 完全一致：置信度 top-K(K=max(match_top_k,
    n_gt)) 候选上做匈牙利指派，代价 = 5*归一化L1坐标距离 - 置信度。
    行 = GT 最近邻间距映射的尺度目标类(E0/E1/E2)，
    列 = 匹配候选在硬路由 gate 上的 argmax 专家。

    ``knn_k``、``scale_centers`` 和 ``scale_sigma_octaves`` 必须来自训练
    criterion，确保 Router graduation 使用与 route loss 相同的标签。
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
):
    """验证集人数 MAE（以所有候选点置信度和作为预测人数）。

    同时评估软路由与硬路由两种模式：
    - soft 使用当前 epoch 的训练温度；
    - hard 固定使用 0.5（hard argmax 不受温度影响）。
    另外复用 hard 前向的预测做 GT 匹配，统计路由混淆矩阵
    （见 _routing_confusion），回答"Router 是否学到尺度分工"。
    """
    model.eval()

    total_abs_error = {"soft": 0.0, "hard": 0.0}
    total_images = 0
    hard_usage = torch.zeros(3, dtype=torch.int64, device=device)
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
                    # 硬路由使用率：仅统计前景候选（置信度>0.5）的 argmax
                    # 分布。全部候选统计会被背景点污染（背景 gate 不受
                    # route 监督，实测漂移到 E2 74% 而真实前景 E2=0%）。
                    fg_mask = scores > 0.5
                    hard_usage += (
                        predictions["gates"][fg_mask]
                        .argmax(dim=-1)
                        .bincount(minlength=3)
                    )

                    # 路由混淆矩阵：复用本次 hard 前向，无额外推理开销
                    batch_confusion, batch_matched = (
                        _routing_confusion(
                            predictions,
                            gt_points,
                            images.shape[-2:],
                            device,
                            match_top_k=match_top_k,
                            knn_k=knn_k,
                            scale_centers=scale_centers,
                            scale_sigma_octaves=scale_sigma_octaves,
                        )
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
    """阶段化温度调度。

    T 不改变 argmax（argmax(z/T) == argmax(z)），降低 T 只会让
    soft gate 更尖锐，不会把错误的 E0 argmax 修成 E1，所以：
    - 软阶段：init-temperature -> phase1-temp（router-grad-epoch
      到达）-> soft-temp-floor（temp-floor-epoch 到达），下限 1.0；
    - hard 阶段：soft-temp-floor -> min-temperature，
      历时 hard-temp-epochs。
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

    if epoch <= args.router_grad_epoch:
        progress = epoch / max(args.router_grad_epoch, 1)
        temperature = (
            args.init_temperature
            - (args.init_temperature - args.phase1_temp)
            * progress
        )
    else:
        span = max(
            args.temp_floor_epoch
            - args.router_grad_epoch,
            1,
        )
        progress = min(
            epoch - args.router_grad_epoch, span
        )
        temperature = (
            args.phase1_temp
            - (args.phase1_temp - args.soft_temp_floor)
            * progress
            / span
        )

    return max(temperature, args.soft_temp_floor)


def router_recalls(
    route_confusion: torch.Tensor,
) -> tuple[list[float], float]:
    """混淆矩阵的行 recall（对角元/行和）与 macro recall。

    recall[i] = 匹配到的 GT E_i 点中，路由到 E_i 的比例。
    """
    row_sums = route_confusion.sum(dim=1)
    recalls = (
        route_confusion.diagonal().float()
        / row_sums.clamp_min(1).float()
    )
    return recalls.tolist(), float(recalls.mean())


def timestamped_save_dir(base: str) -> str:
    """为输出目录附加启动时间戳，避免不同 run 互相覆盖。"""
    return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"


def train_moe(args):
    # 未显式指定输出目录时自动加时间戳；--resume 沿用原 run 目录，
    # 保证恢复训练仍写回同一目录（best_soft/best_hard/last 覆盖逻辑不变）。
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

    graduate_recalls = tuple(
        float(value)
        for value in args.graduate_recalls.split(",")
    )
    if len(graduate_recalls) != 3:
        raise ValueError(
            "--graduate-recalls 需要 3 个值，如 0.60,0.40,0.30"
        )

    # 覆盖保护：从头训练进入已有 save-dir 时备份旧 phase best。
    if not args.resume:
        for best_name in ("best_soft.pt", "best_hard.pt"):
            old_best = os.path.join(args.save_dir, best_name)
            if not os.path.exists(old_best):
                continue
            backup = os.path.join(
                args.save_dir,
                best_name.replace(".pt", "_prev.pt"),
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(f"旧 {best_name} 已备份到 {backup}")

    # 1. 模型
    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        num_references=args.num_references,
    ).to(device)

    # 2. 损失
    criterion = PointMoELoss(
        route_weight=args.route_weight,
        scale_centers=parse_scale_centers(args.scale_centers),
        match_top_k=args.match_top_k,
    )

    # 3. 数据
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

    # 随机裁剪的 GT 分布统计：空 crop 比例过高时分类梯度会压制正样本
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

    # 4. 前几个 epoch 冻结 YOLO 部分，只训练 Point Head
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
    best_hard_selection_score = float("inf")
    hard_started = False
    grad_streak = 0
    first_hard_epoch = None
    last_hard_route = False

    if args.resume and os.path.exists(args.resume):
        logging.info(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as error:
            logging.warning(
                f"旧版 checkpoint 缺少新参数({error})，"
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
            float("inf"),
        )
        best_hard_selection_score = checkpoint.get(
            "best_hard_selection_score",
            float("inf"),
        )
        hard_started = checkpoint.get(
            "hard_started",
            checkpoint.get("epoch", 0) >= 20,
        )
        last_hard_route = checkpoint.get(
            "hard_route",
            hard_started,
        )
        if (
            best_soft_selection_score == float("inf")
            and not last_hard_route
        ):
            best_soft_selection_score = best_selection_score
        if (
            best_hard_selection_score == float("inf")
            and last_hard_route
        ):
            best_hard_selection_score = best_selection_score
        grad_streak = checkpoint.get("grad_streak", 0)
        first_hard_epoch = checkpoint.get(
            "first_hard_epoch",
            checkpoint.get("epoch") if hard_started else None,
        )
        # 覆盖保护：resume 前备份现有 phase best。
        for best_name in ("best_soft.pt", "best_hard.pt"):
            old_best = os.path.join(args.save_dir, best_name)
            if not os.path.exists(old_best):
                continue
            backup = os.path.join(
                args.save_dir,
                best_name.replace(".pt", "_pre_resume.pt"),
            )
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(f"旧 {best_name} 已备份到 {backup}")

        # 恢复后若已超过冻结期，解冻 YOLO（不重建优化器，保留状态）
        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True

    # 5. 训练循环
    # hard 路由由 Router 毕业条件决定（混淆矩阵 recall 连续达标），
    # 不由 epoch 决定；未达标就继续 soft——提前切 hard 等于
    # 正式宣布少数专家死刑。可选 --force-hard-epoch 强制覆盖。
    was_hard_route = bool(last_hard_route)

    for epoch in range(start_epoch, args.epochs):
        hard_route = hard_started
        router_grad = epoch >= args.router_grad_epoch

        if (
            not hard_route
            and args.force_hard_epoch is not None
            and epoch >= args.force_hard_epoch
        ):
            hard_started = True
            hard_route = True
            logging.info(
                f"强制启用硬路由（force-hard-epoch="
                f"{args.force_hard_epoch}）"
            )

        # 软->硬切换：best_hard 基准重置
        if hard_route and not was_hard_route:
            was_hard_route = True
            first_hard_epoch = epoch
            best_selection_score = float("inf")
            best_hard_selection_score = float("inf")
            logging.info(
                "切换硬路由，best_hard 基准重置为 hard MAE"
            )

        temperature = temperature_for_epoch(
            epoch,
            hard_route,
            first_hard_epoch,
            args,
        )

        # 到达解冻点：解冻 YOLO，不重建优化器——
        # 冻结参数在优化器中自动跳过，解冻后自然开始积累 Adam 状态
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
        matched_gate_points = 0
        matched_gate_entropy_sum = torch.zeros(
            (), device=device
        )
        target_mean = torch.zeros(
            3, device=device
        )
        target_points = 0
        loss_sums = {
            name: torch.zeros((), device=device)
            for name in ("cls", "point", "count", "route")
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
                expert_uniform_floor=args.expert_uniform_floor,
            )

            loss, loss_items = criterion(
                predictions,
                gt_points,
                image_size=images.shape[-2:],
            )

            # 诊断只统计 Hungarian 匹配到的正样本，避免背景候选
            # 污染 gate 分布与 Router entropy。
            matched_gate_sum += loss_items[
                "matched_gate_hist"
            ].to(device)
            matched_gate_points += int(
                loss_items["matched_gate_count"].item()
            )
            matched_gate_entropy_sum += loss_items[
                "matched_gate_entropy"
            ].to(device)
            target_mean += loss_items["gate_target_hist"].to(
                device
            )
            target_points += int(
                loss_items["gate_target_count"].item()
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
                loss_sums[loss_name] / max(num_batches, 1)
            ).item()
            for loss_name in loss_sums
        }

        # 6. 验证与保存
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
        )

        gate_pct = (
            matched_gate_sum
            / max(matched_gate_points, 1)
            * 100
        )
        target_pct = target_mean / max(target_points, 1) * 100
        usage_pct = (
            hard_usage.float()
            / max(int(hard_usage.sum()), 1)
            * 100
        )
        router_entropy = (
            matched_gate_entropy_sum
            / max(matched_gate_points, 1)
        ).item()

        logging.info(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} "
            f"cls={avg_loss_items['cls']:.4f} "
            f"point={avg_loss_items['point']:.4f} "
            f"count={avg_loss_items['count']:.4f} "
            f"route={avg_loss_items['route']:.4f} "
            f"T={temperature:.2f} "
            f"hard_route={hard_route} "
            f"router_grad={router_grad} "
            f"soft_MAE={soft_mae:.3f} "
            f"hard_MAE={hard_mae:.3f} "
            f"matched_entropy={router_entropy:.4f}"
        )
        logging.info(
            f"  matched_gate=E0:{gate_pct[0]:.1f}% "
            f"E1:{gate_pct[1]:.1f}% E2:{gate_pct[2]:.1f}%"
            f" | target=E0:{target_pct[0]:.1f}% "
            f"E1:{target_pct[1]:.1f}% E2:{target_pct[2]:.1f}%"
            f" | 硬路由使用=E0:{usage_pct[0]:.1f}% "
            f"E1:{usage_pct[1]:.1f}% E2:{usage_pct[2]:.1f}%"
        )

        # 路由混淆矩阵（匹配正样本口径）：行=GT 尺度目标类，
        # 列=预测专家。主对角线占优说明 Router 学到尺度分工；
        # 某行集中到 E0 说明该尺度的专家在 hard 阶段拿不到样本。
        if matched_points > 0:
            # 0~1 比例乘 100 再拼 %（此前漏乘 100，1.0 显示成 "1.0%"）
            confusion_pct = (
                100.0
                * route_confusion.float()
                / route_confusion.sum(
                    dim=1, keepdim=True
                ).clamp_min(1)
            )
            row_counts = route_confusion.sum(dim=1)
            logging.info(
                "  路由混淆 (行=GT尺度目标类, 列=预测专家):"
            )
            for expert_index in range(3):
                logging.info(
                    f"    GT E{expert_index} -> "
                    f"E0:{confusion_pct[expert_index, 0]:5.1f}% "
                    f"E1:{confusion_pct[expert_index, 1]:5.1f}% "
                    f"E2:{confusion_pct[expert_index, 2]:5.1f}%"
                    f"  (n={int(row_counts[expert_index])})"
                )

        # Router 毕业检查：E0/E1/E2 recall 均达标且 macro recall 达标，
        # 连续 graduate-stable-epochs 轮后才切 hard（下一 epoch 生效）。
        if not hard_started and matched_points > 0:
            recalls, macro_recall = router_recalls(
                route_confusion
            )
            if (
                recalls[0] >= graduate_recalls[0]
                and recalls[1] >= graduate_recalls[1]
                and recalls[2] >= graduate_recalls[2]
                and macro_recall >= args.graduate_macro_recall
            ):
                grad_streak += 1
                if grad_streak >= args.graduate_stable_epochs:
                    hard_started = True
                    first_hard_epoch = epoch + 1
                    logging.info(
                        f"Router 毕业（连续 {grad_streak} 轮达标，"
                        f"recall=E0:{recalls[0]:.2f} "
                        f"E1:{recalls[1]:.2f} "
                        f"E2:{recalls[2]:.2f} "
                        f"macro={macro_recall:.2f}），"
                        "下一 epoch 进入 hard 路由"
                    )
                else:
                    logging.info(
                        f"Router 达标 streak={grad_streak}/"
                        f"{args.graduate_stable_epochs} "
                        f"(recall=E0:{recalls[0]:.2f} "
                        f"E1:{recalls[1]:.2f} "
                        f"E2:{recalls[2]:.2f} "
                        f"macro={macro_recall:.2f})"
                    )
            else:
                grad_streak = 0
                logging.info(
                    f"Router 未达标，继续 soft "
                    f"(recall=E0:{recalls[0]:.2f} "
                    f"E1:{recalls[1]:.2f} "
                    f"E2:{recalls[2]:.2f} "
                    f"macro={macro_recall:.2f})"
                )

        # best 选择：soft/hard 分开保存，避免 hard 阶段覆盖 soft 证据。
        if hard_route:
            val_score_for_best = hard_mae
            metric_name = "hard MAE"
            active_best_score = best_hard_selection_score
        else:
            val_score_for_best = soft_mae
            metric_name = "soft MAE"
            active_best_score = best_soft_selection_score

        improved = val_score_for_best < active_best_score
        if improved:
            best_selection_score = val_score_for_best
            if hard_route:
                best_hard_selection_score = val_score_for_best
                best_path = os.path.join(
                    args.save_dir, "best_hard.pt"
                )
            else:
                best_soft_selection_score = val_score_for_best
                best_path = os.path.join(
                    args.save_dir, "best_soft.pt"
                )
        else:
            best_selection_score = active_best_score
            best_path = None

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_selection_score,
            "best_selection_score": best_selection_score,
            "best_soft_selection_score": best_soft_selection_score,
            "best_hard_selection_score": best_hard_selection_score,
            "selection_metric": metric_name,
            "hard_started": hard_started,
            "hard_route": hard_route,
            "grad_streak": grad_streak,
            "first_hard_epoch": first_hard_epoch,
            "soft_MAE": soft_mae,
            "hard_MAE": hard_mae,
            "val_hard_expert_usage": hard_usage.detach().cpu(),
            "router_confusion": route_confusion.detach().cpu(),
            "router_recalls": (
                router_recalls(route_confusion)[0]
                if matched_points > 0
                else [0.0, 0.0, 0.0]
            ),
            "router_entropy": router_entropy,
            "args": vars(args),
            "config": build_checkpoint_config(
                args,
                criterion,
                hard_started=hard_started,
                hard_route=hard_route,
                temperature=temperature,
            ),
        }
        if improved:
            torch.save(checkpoint_data, best_path)
            logging.info(
                f"  -> 新的最佳 {metric_name}: {best_selection_score:.3f}，"
                f"已保存 {best_path}"
            )

        torch.save(
            checkpoint_data,
            os.path.join(args.save_dir, "last.pt"),
        )

    if not hard_started:
        logging.warning(
            "Router never graduated to hard routing; "
            "best_hard.pt was not produced."
        )
    logging.info("训练结束。")


def build_parser():
    parser = argparse.ArgumentParser(
        description="训练 YOLO11 + 点级 Scale-MoE Head 人群计数模型"
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11n.pt",
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
        "--hidden-channels", type=int, default=128
    )
    parser.add_argument(
        "--num-references", type=int, default=4,
        help="每个网格参考点数 K（1/4/9）"
    )
    parser.add_argument(
        "--scale-centers",
        type=str,
        default="10,20,40",
        help="Router 三个尺度中心（像素），默认 10,20,40",
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
        help="软阶段起始温度"
    )
    parser.add_argument(
        "--min-temperature", type=float, default=0.5,
        help="hard 阶段温度下限"
    )
    parser.add_argument(
        "--phase1-temp", type=float, default=1.3,
        help="router-grad-epoch 时的温度（epoch 0 到该点线性插值）"
    )
    parser.add_argument(
        "--soft-temp-floor", type=float, default=1.0,
        help="hard 切换前的温度下限。T 不改变 argmax，"
        "降到 0.5 只会让 soft gate 更尖锐，无助于 Router 学尺度"
    )
    parser.add_argument(
        "--temp-floor-epoch", type=int, default=30,
        help="软阶段温度降到 soft-temp-floor 的 epoch"
    )
    parser.add_argument(
        "--hard-temp-epochs", type=int, default=20,
        help="hard 阶段温度从 soft-temp-floor 衰减到 "
        "min-temperature 的 epoch 数"
    )
    parser.add_argument(
        "--router-grad-epoch", type=int, default=15,
        help="从该 epoch 起允许 cls/point/count 梯度流入 Router；"
        "之前 Router 只接受 balanced L_route（梯度隔离）"
    )
    parser.add_argument(
        "--expert-uniform-floor",
        type=float,
        default=0.3,
        help="Router warm-up 时每个专家获得的最小 task gradient 比例",
    )
    parser.add_argument(
        "--force-hard-epoch", type=int, default=None,
        help="可选：强制在该 epoch 启用硬路由，绕过毕业条件"
    )
    parser.add_argument(
        "--graduate-recalls", type=str,
        default="0.60,0.40,0.30",
        help="Router 毕业条件：混淆矩阵上 E0/E1/E2 的最小 recall"
    )
    parser.add_argument(
        "--graduate-macro-recall", type=float, default=0.50,
        help="Router 毕业条件：macro recall 下限"
    )
    parser.add_argument(
        "--graduate-stable-epochs", type=int, default=3,
        help="连续满足毕业条件的 epoch 数后才切 hard"
    )
    parser.add_argument(
        "--route-weight", type=float, default=0.15,
        help="尺度路由监督(macro CE)损失权重"
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
        help="从指定 checkpoint 恢复训练"
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
