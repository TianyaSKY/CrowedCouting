import argparse
import logging
import math
import os
import shutil
import time

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.point_moe_loss import PointMoELoss
from models.yolo11_moe_point import YOLO11MoEPoint
from scripts.data.point_dataset import PointDataset, point_collate_fn
from scripts.visualization.validation_visualizer import (
    log_validation_images,
)


NATIVE_ARCHITECTURE = "native_multiscale"


def setup_logging(log_path: str) -> None:
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
            f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
            f"支持的架构: {supported}。"
        )

    try:
        torch.zeros(1, device="cuda").sum().item()
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError(
            "CUDA 设备初始化/基础算子探测失败；请检查 NVIDIA 驱动与 PyTorch CUDA 版本。"
        ) from error


def parse_native_references(value: str) -> tuple[int, int, int]:
    references = tuple(int(item.strip()) for item in value.split(","))
    if len(references) != 3 or any(reference <= 0 for reference in references):
        raise ValueError(
            "--native-references 需要三个正整数，例如 1,4,16"
        )
    for reference in references:
        side = math.isqrt(reference)
        if side * side != reference:
            raise ValueError(
                "--native-references 的每个值必须是完全平方数"
            )
    return references[0], references[1], references[2]


def native_matching_mode(epoch: int, args) -> str:
    return (
        "independent"
        if epoch < args.native_warmup_epochs
        else "competitive"
    )


def native_matching_stage(epoch: int, args) -> str:
    return (
        "warmup_independent"
        if epoch < args.native_warmup_epochs
        else "competitive_global"
    )


def build_optimizer(model, args):
    return torch.optim.AdamW(
        [
            {
                "params": list(model.point_head.parameters()),
                "lr": args.head_lr,
            },
            {
                "params": list(model.yolo.parameters()),
                "lr": args.backbone_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )


def build_checkpoint_config(
    args,
    criterion: PointMoELoss,
    *,
    matching_stage: str,
) -> dict[str, object]:
    return {
        "checkpoint_version": 2,
        "architecture": NATIVE_ARCHITECTURE,
        "crop_size": int(args.crop_size),
        "hidden_channels": int(args.hidden_channels),
        "native_references": list(
            parse_native_references(args.native_references)
        ),
        "native_warmup_epochs": int(args.native_warmup_epochs),
        "matching_stage": matching_stage,
        "matching_schedule": (
            "independent_per_expert_then_global_hungarian"
        ),
        "selection_metric": "native weighted normalized MAE",
        "match_top_k": int(criterion.match_top_k),
        "match_position_weight": float(
            criterion.match_position_weight
        ),
        "match_confidence_weight": float(
            criterion.match_confidence_weight
        ),
        "candidate_preselection": "expert_balanced_top_k",
    }


def timestamped_save_dir(base: str) -> str:
    return f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"


def dataset_mean_gt_count(dataset) -> float:
    total_points = 0
    for image_path in dataset.image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        point_path = os.path.join(
            dataset.points_dir,
            base_name + ".txt",
        )
        if not os.path.exists(point_path):
            continue
        with open(point_path, encoding="utf-8") as point_file:
            total_points += sum(
                len(line.split()) >= 2 for line in point_file
            )
    return total_points / max(len(dataset), 1)


def validation_image_options(args) -> tuple[int, int, float]:
    interval = int(getattr(args, "val_image_interval", 1))
    count = int(getattr(args, "val_image_count", 4))
    conf_threshold = float(getattr(args, "val_image_conf", 0.5))
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
    writer.add_scalar("system/epoch_seconds", epoch_seconds, epoch)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    else:
        allocated_mb = 0.0
        reserved_mb = 0.0
    writer.add_scalar(
        "system/gpu_memory_allocated_mb", allocated_mb, epoch
    )
    writer.add_scalar(
        "system/gpu_memory_reserved_mb", reserved_mb, epoch
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


def evaluate_native_count_mae(
    model,
    val_loader,
    device,
    criterion: PointMoELoss | None = None,
    max_visual_samples: int = 0,
):
    if max_visual_samples < 0:
        raise ValueError("max_visual_samples must not be negative")

    was_training = model.training
    model.eval()
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_bias = 0.0
    total_images = 0
    num_batches = 0
    loss_sums = (
        {
            name: torch.zeros((), device=device)
            for name in ("total", "cls", "point", "count")
        }
        if criterion is not None
        else None
    )
    winner_hist = torch.zeros(3, dtype=torch.int64, device=device)
    positive_count = torch.zeros(3, device=device)
    distance_sum = torch.zeros(3, device=device)
    confidence_sum = torch.zeros(3, device=device)
    matched_count = 0
    validation_samples: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Native 验证中", leave=False):
            images = batch["img"].to(device)
            gt_points = [p.to(device) for p in batch["points"]]
            predictions = model(images)
            pred_counts = predictions["logits"].sigmoid().sum(dim=1)

            for image_index, gt in enumerate(gt_points):
                error = float(pred_counts[image_index].item()) - gt.shape[0]
                total_abs_error += abs(error)
                total_squared_error += error * error
                total_bias += error

            if criterion is not None:
                native_loss, loss_items = criterion(
                    predictions,
                    gt_points,
                    image_size=images.shape[-2:],
                    matching_mode="competitive",
                )
                loss_sums["total"] += native_loss.detach()
                for name in ("cls", "point", "count"):
                    loss_sums[name] += loss_items[name].detach()
                winner_hist += loss_items["winner_hist"].to(
                    device=device,
                    dtype=torch.int64,
                )
                positive_count += loss_items["positive_count"].to(device)
                distance_sum += loss_items["matched_distance_sum"].to(device)
                confidence_sum += loss_items[
                    "matched_confidence_sum"
                ].to(device)
                matched_count += int(loss_items["matched_count"].item())

            if len(validation_samples) < max_visual_samples:
                image_paths = batch.get("image_paths", [])
                for image_index in range(len(gt_points)):
                    if len(validation_samples) >= max_visual_samples:
                        break
                    image_path = None
                    if isinstance(image_paths, list):
                        image_path = image_paths[image_index]
                    validation_samples.append(
                        {
                            "image": images[image_index].detach().cpu(),
                            "gt_points": gt_points[
                                image_index
                            ].detach().cpu(),
                            "predictions": {
                                name: predictions[name][image_index]
                                .detach()
                                .cpu()
                                for name in (
                                    "logits",
                                    "points",
                                    "expert_indices",
                                )
                            },
                            "image_path": image_path,
                        }
                    )

            total_images += len(gt_points)
            num_batches += 1

    if was_training:
        model.train()

    result = {
        "mae": total_abs_error / max(total_images, 1),
        "rmse": (total_squared_error / max(total_images, 1)) ** 0.5,
        "bias": total_bias / max(total_images, 1),
        "winner_hist": winner_hist,
        "positive_count": positive_count,
        "matched_distance_sum": distance_sum,
        "matched_confidence_sum": confidence_sum,
        "matched_count": matched_count,
        "validation_samples": validation_samples,
    }
    if loss_sums is None:
        result["loss"] = None
    else:
        result["loss"] = {
            name: (value / max(num_batches, 1)).item()
            for name, value in loss_sums.items()
        }
    return result


def evaluate_expert_only_mae(model, val_loader, device):
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
            gt_counts = [points.shape[0] for points in batch["points"]]
            for expert_index in range(3):
                predictions = model(
                    images,
                    routing_mode="expert_only",
                    expert_index=expert_index,
                )
                pred_counts = predictions["logits"].sigmoid().sum(dim=1)
                for image_index, gt_count in enumerate(gt_counts):
                    total_abs_error[expert_index] += abs(
                        float(pred_counts[image_index].item()) - gt_count
                    )
            total_images += len(gt_counts)

    if was_training:
        model.train()
    return {
        f"E{expert_index}": total_abs_error[expert_index]
        / max(total_images, 1)
        for expert_index in range(3)
    }


def train_moe(args):
    if args.native_warmup_epochs < 0:
        raise ValueError("--native-warmup-epochs 不能为负数")
    native_references = parse_native_references(args.native_references)

    if args.save_dir is None:
        if args.resume:
            args.save_dir = os.path.dirname(os.path.abspath(args.resume))
        else:
            args.save_dir = timestamped_save_dir("runs/native_multiscale")

    os.makedirs(args.save_dir, exist_ok=True)
    setup_logging(os.path.join(args.save_dir, "train.log"))
    logging.info("输出目录: %s", args.save_dir)

    writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "tensorboard")
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)
    validate_cuda_device(device)

    if not args.resume:
        old_best = os.path.join(args.save_dir, "best_native.pt")
        if os.path.exists(old_best):
            backup = os.path.join(args.save_dir, "best_native_prev.pt")
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(
                    "旧 best_native.pt 已备份到 %s",
                    backup,
                )
    criterion = PointMoELoss(
        match_top_k=args.match_top_k,
        match_position_weight=args.match_position_weight,
        match_confidence_weight=args.match_confidence_weight,
    )

    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        native_references=native_references,
    ).to(device)

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

    gt_counts = train_dataset.sample_gt_counts(num_samples=100)
    if gt_counts.size > 0:
        logging.info(
            "训练裁剪 GT 统计(采样 %d 张): mean=%.1f median=%.0f max=%d zero_ratio=%.1f%%",
            gt_counts.size,
            float(gt_counts.mean()),
            float(np.median(gt_counts)),
            int(gt_counts.max()),
            float((gt_counts == 0).mean() * 100),
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
            "前 %d 个 epoch 冻结 YOLO Backbone+Neck",
            args.freeze_epochs,
        )
        for param in model.yolo.parameters():
            param.requires_grad = False

    optimizer = build_optimizer(model, args)
    start_epoch = 0
    best_selection_score = float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint.get("config", {})
        if not isinstance(config, dict) or config.get("architecture") != NATIVE_ARCHITECTURE:
            raise ValueError(
                "只支持 native_multiscale checkpoint；旧 D2 checkpoint 已删除"
            )
        saved_refs = tuple(config.get("native_references", ()))
        if saved_refs != native_references:
            raise ValueError(
                "resume checkpoint 的 native_references 与当前配置不一致"
            )
        saved_position_weight = float(
            config.get("match_position_weight", 5.0)
        )
        saved_confidence_weight = float(
            config.get("match_confidence_weight", 0.25)
        )
        if not math.isclose(
            saved_position_weight,
            args.match_position_weight,
        ) or not math.isclose(
            saved_confidence_weight,
            args.match_confidence_weight,
        ):
            raise ValueError(
                "resume checkpoint 的 matching cost 权重与当前配置不一致"
            )
        model.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, RuntimeError) as error:
            logging.warning("优化器状态不兼容，使用全新优化器: %s", error)
            optimizer = build_optimizer(model, args)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_selection_score = float(
            checkpoint.get("best_selection_score", float("inf"))
        )
        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            logging.info(
                "resume start_epoch=%d：YOLO Backbone+Neck 保持可训练",
                start_epoch,
            )

        old_best = os.path.join(args.save_dir, "best_native.pt")
        if os.path.exists(old_best):
            backup = os.path.join(args.save_dir, "best_native_pre_resume.pt")
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)

    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.perf_counter()
        matching_mode = native_matching_mode(epoch, args)
        matching_stage = native_matching_stage(epoch, args)
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            logging.info("解冻 YOLO Backbone+Neck")
        if epoch == args.native_warmup_epochs:
            logging.info("启用 native_multiscale global Hungarian competition")

        model.train()
        total_loss = 0.0
        num_batches = 0
        loss_sums = {
            name: torch.zeros((), device=device)
            for name in ("cls", "point", "count")
        }
        train_winner_sum = torch.zeros(
            3, dtype=torch.int64, device=device
        )
        train_positive_sum = torch.zeros(3, device=device)
        train_distance_sum = torch.zeros(3, device=device)
        train_confidence_sum = torch.zeros(3, device=device)
        train_matched_count = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            images = batch["img"].to(device)
            gt_points = [p.to(device) for p in batch["points"]]
            predictions = model(images)
            loss, loss_items = criterion(
                predictions,
                gt_points,
                image_size=images.shape[-2:],
                matching_mode=matching_mode,
            )

            train_winner_sum += loss_items["winner_hist"].to(
                device=device,
                dtype=torch.int64,
            )
            train_positive_sum += loss_items["positive_count"].to(device)
            train_distance_sum += loss_items["matched_distance_sum"].to(device)
            train_confidence_sum += loss_items[
                "matched_confidence_sum"
            ].to(device)
            train_matched_count += int(loss_items["matched_count"].item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )
            optimizer.step()

            total_loss += loss.item()
            for name in loss_sums:
                loss_sums[name] += loss_items[name].detach()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_items = {
            name: (value / max(num_batches, 1)).item()
            for name, value in loss_sums.items()
        }
        collect_visuals = (
            val_image_interval > 0
            and (epoch + 1) % val_image_interval == 0
        )
        validation = evaluate_native_count_mae(
            model,
            val_loader,
            device,
            criterion=criterion,
            max_visual_samples=val_image_count if collect_visuals else 0,
        )

        native_mae = validation["mae"]
        native_rmse = validation["rmse"]
        native_bias = validation["bias"]
        native_norm_mae = native_mae / max(val_mean_gt_count, 1e-12)
        val_winner_hist = validation["winner_hist"]
        val_positive_count = validation["positive_count"]
        val_distance_mean = validation["matched_distance_sum"] / val_positive_count.clamp_min(1)
        val_confidence_mean = validation["matched_confidence_sum"] / val_positive_count.clamp_min(1)
        train_distance_mean = train_distance_sum / train_positive_sum.clamp_min(1)
        train_confidence_mean = train_confidence_sum / train_positive_sum.clamp_min(1)
        val_winner_pct = val_winner_hist.float() / max(int(val_winner_hist.sum()), 1) * 100
        train_winner_pct = train_winner_sum.float() / max(int(train_winner_sum.sum()), 1) * 100

        logging.info(
            "[Epoch %d/%d] loss=%.4f cls=%.4f point=%.4f count=%.4f stage=%s MAE=%.3f/%.6f RMSE=%.3f bias=%.3f",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_loss_items["cls"],
            avg_loss_items["point"],
            avg_loss_items["count"],
            matching_stage,
            native_mae,
            native_norm_mae,
            native_rmse,
            native_bias,
        )
        logging.info(
            "  GT winner: E0=%.1f%% E1=%.1f%% E2=%.1f%%",
            val_winner_pct[0],
            val_winner_pct[1],
            val_winner_pct[2],
        )
        logging.info(
            "  train GT winner: E0=%.1f%% E1=%.1f%% E2=%.1f%%",
            train_winner_pct[0],
            train_winner_pct[1],
            train_winner_pct[2],
        )
        logging.info(
            "  matched distance(px): E0=%.2f E1=%.2f E2=%.2f",
            val_distance_mean[0],
            val_distance_mean[1],
            val_distance_mean[2],
        )
        logging.info(
            "  matched confidence: E0=%.3f E1=%.3f E2=%.3f",
            val_confidence_mean[0],
            val_confidence_mean[1],
            val_confidence_mean[2],
        )
        logging.info(
            "  positive count: E0=%d E1=%d E2=%d | train: E0=%.1f E1=%.1f E2=%.1f",
            int(val_positive_count[0]),
            int(val_positive_count[1]),
            int(val_positive_count[2]),
            train_positive_sum[0],
            train_positive_sum[1],
            train_positive_sum[2],
        )
        if validation["loss"] is not None:
            logging.info("  validation loss: %s", validation["loss"])

        for name, value in (
            ("total", avg_loss),
            ("cls", avg_loss_items["cls"]),
            ("point", avg_loss_items["point"]),
            ("count", avg_loss_items["count"]),
        ):
            writer.add_scalar(f"loss/{name}", value, epoch)
            writer.add_scalar(f"train/loss_{name}", value, epoch)
        if validation["loss"] is not None:
            for name, value in validation["loss"].items():
                writer.add_scalar(f"val/loss_native_{name}", value, epoch)
        writer.add_scalar("mae/native_raw", native_mae, epoch)
        writer.add_scalar("mae/native_weighted_norm", native_norm_mae, epoch)
        writer.add_scalar("val/rmse_native", native_rmse, epoch)
        writer.add_scalar("val/count_bias_native", native_bias, epoch)
        writer.add_scalar("native/matched_count", validation["matched_count"], epoch)
        writer.add_scalar("schedule/native_warmup", float(matching_mode == "independent"), epoch)
        for expert_index in range(3):
            writer.add_scalar(f"native/winner_E{expert_index}_pct", float(val_winner_pct[expert_index]), epoch)
            writer.add_scalar(f"native/train_winner_E{expert_index}_pct", float(train_winner_pct[expert_index]), epoch)
            writer.add_scalar(f"native/matched_distance_E{expert_index}", float(val_distance_mean[expert_index]), epoch)
            writer.add_scalar(f"native/matched_confidence_E{expert_index}", float(val_confidence_mean[expert_index]), epoch)
            writer.add_scalar(f"native/positive_count_E{expert_index}", float(val_positive_count[expert_index]), epoch)
            writer.add_scalar(f"native/train_positive_count_E{expert_index}", float(train_positive_sum[expert_index]), epoch)
            writer.add_scalar(f"native/train_matched_distance_E{expert_index}", float(train_distance_mean[expert_index]), epoch)
            writer.add_scalar(f"native/train_matched_confidence_E{expert_index}", float(train_confidence_mean[expert_index]), epoch)

        if (
            matching_mode == "independent"
            and args.expert_only_eval_interval > 0
            and (epoch + 1) % args.expert_only_eval_interval == 0
        ):
            expert_only_mae = evaluate_expert_only_mae(
                model,
                val_loader,
                device,
            )
            logging.info("  expert-only MAE: %s", expert_only_mae)
            for expert_index in range(3):
                writer.add_scalar(
                    f"mae/expert_only_E{expert_index}",
                    expert_only_mae[f"E{expert_index}"],
                    epoch,
                )
        else:
            expert_only_mae = None

        if collect_visuals and validation["validation_samples"]:
            log_validation_images(
                writer,
                f"val_images/{dataset_tag}",
                validation["validation_samples"],
                epoch,
                conf_threshold=val_image_conf,
            )

        improved = (
            matching_mode == "competitive"
            and native_norm_mae < best_selection_score
        )
        if improved:
            best_selection_score = native_norm_mae
        if math.isfinite(best_selection_score):
            writer.add_scalar("val/best_score", best_selection_score, epoch)

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_selection_score": best_selection_score,
            "best_mae": best_selection_score,
            "selection_metric": "native weighted normalized MAE",
            "architecture": NATIVE_ARCHITECTURE,
            "native_mae": native_mae,
            "native_rmse": native_rmse,
            "native_bias": native_bias,
            "native_norm_mae": native_norm_mae,
            "matching_stage": matching_stage,
            "native_winner_hist": val_winner_hist.detach().cpu(),
            "native_train_winner_hist": train_winner_sum.detach().cpu(),
            "native_positive_count": val_positive_count.detach().cpu(),
            "native_matched_distance_mean": val_distance_mean.detach().cpu(),
            "native_matched_confidence_mean": val_confidence_mean.detach().cpu(),
            "native_matched_count": validation["matched_count"],
            "expert_only_mae": expert_only_mae,
            "args": vars(args),
            "config": build_checkpoint_config(
                args,
                criterion,
                matching_stage=matching_stage,
            ),
        }
        if improved:
            best_path = os.path.join(args.save_dir, "best_native.pt")
            torch.save(checkpoint_data, best_path)
            logging.info(
                "  -> 新的最佳 native weighted normalized MAE: %.6f (%s)",
                best_selection_score,
                best_path,
            )
        torch.save(checkpoint_data, os.path.join(args.save_dir, "last.pt"))
        log_system_metrics(
            writer,
            optimizer,
            epoch,
            time.perf_counter() - epoch_started_at,
            device,
        )
        writer.flush()

    writer.close()
    logging.info("训练结束。")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "训练 YOLO11 native_multiscale P3/P4/P5 point experts；"
            "warmup 独立 matching，之后全局 Hungarian competition"
        )
    )
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/shanghaitech_AB",
    )
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument(
        "--native-references",
        type=str,
        default="1,4,16",
        help="P3/P4/P5 的 K，默认 1,4,16",
    )
    parser.add_argument(
        "--native-warmup-epochs",
        type=int,
        default=5,
        help="前 N 个 epoch 三个 Expert 独立 matching",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--expert-only-eval-interval",
        type=int,
        default=5,
        help="每 N 个 warmup epoch 运行 E0/E1/E2-only 诊断；competition 阶段关闭",
    )
    parser.add_argument(
        "--val-image-interval",
        type=int,
        default=1,
        help="每 N 个 epoch 写入验证图；0 关闭",
    )
    parser.add_argument("--val-image-count", type=int, default=4)
    parser.add_argument("--val-image-conf", type=float, default=0.5)
    parser.add_argument("--match-top-k", type=int, default=2000)
    parser.add_argument(
        "--match-position-weight",
        type=float,
        default=5.0,
        help="Hungarian cost 的位置误差权重",
    )
    parser.add_argument(
        "--match-confidence-weight",
        type=float,
        default=0.25,
        help="Hungarian cost 的 confidence 权重",
    )
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="仅从 native_multiscale checkpoint 恢复",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
