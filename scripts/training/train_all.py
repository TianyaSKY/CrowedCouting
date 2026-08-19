"""在所有数据集上联合训练（batch 混合）+ 逐数据集验证。

训练数据 = 各数据集 train split 的 ConcatDataset（按图片自然采样），
验证 = 每个数据集各自跑 soft/hard MAE；hard weighted normalized MAE
作为主选模指标，macro normalized MAE 仅作为跨数据集泛化参考。官方
test 只允许在训练结束后评估。

用法（GPU 机器，从项目根目录）:

    python -m scripts.training.train_all \
        --weights yolo11m.pt \
        --save-dir runs/moe_point_all \
        --epochs 100

默认数据集（--dataset 可覆盖，可重复）:
    shanghaitech=datasets/shanghaitech_AB:train:val
    jhu=datasets/jhu_crowd:train:val
    qnrf=datasets/ucf_qnrf:train:val
    cc50=datasets/ucf_cc50:fold0_train:fold0_val

--dataset 格式: name=root:train_split[:train_split2...]:eval_split
（train split 用 + 连接多个；CC-50 默认只训练 fold0 的 36 张，
 验证用同一 outer fold 的 fold0_val 留出 4 张。）

超参数与 train_moe 完全一致（共享 argparse），checkpoint 格式兼容
evaluate_datasets / test_each_dataset。先转换数据:

    python -m scripts.data.prepare_all
"""

import argparse
import glob
import os
import shutil

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import logging

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.point_moe_loss import PointMoELoss
from models.yolo11_moe_point import YOLO11MoEPoint
from scripts.data.point_dataset import PointDataset, point_collate_fn
from scripts.training import train_moe as tm

DEFAULT_DATASETS = [
    "shanghaitech=datasets/shanghaitech_AB:train:val",
    "jhu=datasets/jhu_crowd:train:val",
    "qnrf=datasets/ucf_qnrf:train:val",
    "cc50=datasets/ucf_cc50:fold0_train:fold0_val",
]


def parse_dataset_spec(spec: str) -> tuple[str, str, list[str], str]:
    """解析 'name=root:train[:train2...]:eval'。"""
    name, _, rest = spec.partition("=")
    root, _, splits = rest.partition(":")
    train_part, _, eval_split = splits.partition(":")
    train_splits = [s for s in train_part.split("+") if s]
    if not (name and root and train_splits and eval_split):
        raise SystemExit(
            f"--dataset 格式应为 name=root:train[:train2...]:eval，"
            f"收到: {spec}"
        )
    return name, root, train_splits, eval_split


def assert_disjoint_splits(
    root: str,
    train_splits: list[str],
    eval_split: str,
) -> None:
    """确认训练 split 与训练期评估 split 没有同名图片。"""
    train_names = set()

    for split in train_splits:
        paths = glob.glob(
            os.path.join(root, "images", split, "*.jpg")
        )
        train_names.update(
            os.path.basename(path) for path in paths
        )

    eval_paths = glob.glob(
        os.path.join(root, "images", eval_split, "*.jpg")
    )
    eval_names = {
        os.path.basename(path) for path in eval_paths
    }

    overlap = train_names & eval_names

    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(
            f"检测到 train/eval 数据泄漏: "
            f"{len(overlap)} 张重复，例如 {examples}"
        )


def reject_test_as_validation(eval_split: str) -> None:
    """默认禁止用官方 test split 参与训练期选模。"""
    if "test" in eval_split.lower():
        raise ValueError(
            f"训练期间禁止使用 test split 选模: {eval_split}"
        )


def reject_test_as_training(train_splits: list[str]) -> None:
    """官方 test split 不得通过训练 split 进入反向传播。"""
    invalid = [
        split for split in train_splits
        if "test" in split.lower()
    ]
    if invalid:
        raise ValueError(
            "训练期间禁止使用 test split: "
            f"{invalid}"
        )


def reject_multiple_outer_fold_train_splits(
    train_splits: list[str],
) -> None:
    """禁止把不同 CC50 outer fold 的 train 拼成一个训练集。"""
    fold_train_splits = [
        split for split in train_splits
        if split.lower().startswith("fold")
        and split.lower().endswith("_train")
    ]
    if len(fold_train_splits) > 1:
        raise ValueError(
            "CC50 每次只能训练一个 outer fold，禁止拼接: "
            f"{fold_train_splits}"
        )


def mean_gt_count(dataset: PointDataset) -> float:
    """读取验证集平均 GT 人数，用于归一化 MAE 选模。"""
    total_points = 0
    for image_path in dataset.image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        point_path = os.path.join(
            dataset.points_dir,
            base_name + ".txt",
        )
        if not os.path.exists(point_path):
            continue

        with open(point_path, "r") as point_file:
            for line in point_file:
                if len(line.split()) >= 2:
                    total_points += 1

    mean_count = total_points / max(len(dataset), 1)
    if mean_count <= 0:
        raise ValueError(
            f"验证集 {dataset.image_dir} 的平均 GT 人数为 0，"
            "无法计算 normalized MAE"
        )
    return mean_count



def weighted_normalized_mae(
    per_dataset: dict[str, dict[str, float]],
    val_mean_gt_counts: dict[str, float],
    val_image_counts: dict[str, int],
    mode: str,
) -> float:
    """Normalize each dataset MAE, then weight by validation image count."""
    total_images = sum(
        val_image_counts[name] for name in per_dataset
    )
    if total_images <= 0:
        return float("inf")
    return sum(
        val_image_counts[name]
        * per_dataset[name][mode]
        / val_mean_gt_counts[name]
        for name in per_dataset
    ) / total_images


def train_all(args: argparse.Namespace) -> None:
    if args.router_warmup_epochs < 0:
        raise ValueError("--router-warmup-epochs 不能为负数")

    # 未显式指定输出目录时自动加时间戳；--resume 沿用原 run 目录。
    if args.save_dir is None:
        if args.resume:
            args.save_dir = os.path.dirname(
                os.path.abspath(args.resume)
            )
        else:
            args.save_dir = tm.timestamped_save_dir(
                "runs/moe_point_all"
            )

    os.makedirs(args.save_dir, exist_ok=True)
    tm.setup_logging(os.path.join(args.save_dir, "train.log"))
    logging.info("输出目录: %s", args.save_dir)

    # TensorBoard
    tb_dir = os.path.join(args.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info("TensorBoard 日志目录: %s", tb_dir)

    if args.resume:
        logging.warning(
            "resume 仅用于调试；正式 H0 实验应从 yolo11m.pt "
            "新开 run，禁止从 Unsup-v0 soft-only checkpoint resume"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)
    tm.validate_cuda_device(device)

    specs = (
        [parse_dataset_spec(s) for s in args.dataset]
        if args.dataset
        else [parse_dataset_spec(s) for s in DEFAULT_DATASETS]
    )
    allow_test_as_eval = getattr(
        args, "allow_test_as_eval", False
    )
    for name, root, train_splits, eval_split in specs:
        reject_test_as_training(train_splits)
        reject_multiple_outer_fold_train_splits(
            train_splits
        )
        if not allow_test_as_eval:
            reject_test_as_validation(eval_split)
        assert_disjoint_splits(
            root, train_splits, eval_split
        )

    logging.info("联合数据集:")
    for name, root, train_splits, eval_split in specs:
        logging.info(
            "  %s: train=%s (%s) eval=%s",
            name,
            train_splits,
            root,
            eval_split,
        )
        if allow_test_as_eval and "test" in eval_split.lower():
            logging.warning(
                "显式允许 test 参与训练期评估: %s (%s)",
                name,
                eval_split,
            )

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
        scale_centers=tm.parse_scale_centers(
            args.scale_centers
        ),
        match_top_k=args.match_top_k,
    )

    train_datasets = []
    train_dataset_sizes: dict[str, int] = {}
    for name, root, train_splits, eval_split in specs:
        for split in train_splits:
            dataset = PointDataset(
                root,
                split=split,
                crop_size=args.crop_size,
                augment=True,
            )
            train_datasets.append(dataset)
            train_dataset_sizes[name] = (
                train_dataset_sizes.get(name, 0)
                + len(dataset)
            )
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=point_collate_fn,
        drop_last=True,
    )

    val_loaders = {}
    val_mean_gt_counts = {}
    val_image_counts: dict[str, int] = {}
    for name, root, train_splits, eval_split in specs:
        val_dataset = PointDataset(
            root,
            split=eval_split,
            crop_size=args.crop_size,
            augment=False,
        )
        val_mean_gt_counts[name] = mean_gt_count(
            val_dataset
        )
        val_image_counts[name] = len(val_dataset)
        val_loaders[name] = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=point_collate_fn,
        )
        logging.info(
            "  %s 验证集: %d images，平均 GT 人数: %.3f",
            name,
            val_image_counts[name],
            val_mean_gt_counts[name],
        )
    logging.info(
        "训练样本总数: %d（%d 个数据集 split 拼接）",
        len(train_dataset),
        len(train_datasets),
    )
    logging.info(
        "训练按图片自然采样: %s",
        ", ".join(
            f"{name}={count} "
            f"({count / max(len(train_dataset), 1):.1%})"
            for name, count in train_dataset_sizes.items()
        ),
    )

    if args.freeze_epochs > 0:
        logging.info(
            f"前 {args.freeze_epochs} 个 epoch 冻结 YOLO Backbone+Neck"
        )
        for param in model.yolo.parameters():
            param.requires_grad = False

    optimizer = tm.build_optimizer(model, args)
    start_epoch = 0
    best_selection_score = float("inf")

    for param in model.point_head.router.parameters():
        param.requires_grad = False

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
                f"优化器状态不兼容({error})，使用全新优化器"
            )
            optimizer = tm.build_optimizer(model, args)
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
        router_warmup = (
            epoch < args.router_warmup_epochs
        )
        router_active = not router_warmup
        router_grad = router_active
        routing_mode = "train_drop1"
        temperature = tm.temperature_for_epoch(
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
                "解冻 YOLO Backbone+Neck（保留优化器状态）"
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
        matched_gate_points = 0
        train_sampled_usage = torch.zeros(
            3, dtype=torch.int64, device=device
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
            gt_points = [p.to(device) for p in batch["points"]]

            predictions = model(
                images,
                temperature=temperature,
                routing_mode=routing_mode,
                router_grad=router_grad,
            )
            batch_usage, _, _, _ = tm.routing_statistics(
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
                model.parameters(), max_norm=args.grad_clip
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
            name: (
                loss_sums[name] / max(num_batches, 1)
            ).item()
            for name in loss_sums
        }

        per_dataset: dict[str, dict[str, float]] = {}
        expert_only_by_dataset = {}
        full3_top1_usage_sum = torch.zeros(
            3, dtype=torch.int64, device=device
        )
        confusion_sum = torch.zeros(
            3, 3, dtype=torch.int64, device=device
        )
        matched_sum = 0
        val_gate_entropy_sum = torch.zeros(
            (), device=device
        )
        val_gate_margin_sum = torch.zeros(
            (), device=device
        )
        val_gate_points = 0
        for name, val_loader in val_loaders.items():
            validation = tm.evaluate_count_mae(
                model,
                val_loader,
                device,
                temperature=temperature,
                knn_k=criterion.knn_k,
                scale_centers=criterion.scale_centers,
                scale_sigma_octaves=criterion.scale_sigma_octaves,
                diagnose_scale_routing=(
                    args.diagnose_scale_routing
                ),
            )
            per_dataset[name] = validation["mae"]
            full3_top1_usage_sum += validation[
                "full3_top1_usage"
            ]
            val_gate_entropy_sum += validation[
                "full3_gate_entropy_sum"
            ]
            val_gate_margin_sum += validation[
                "full3_gate_margin_sum"
            ]
            val_gate_points += validation[
                "full3_gate_points"
            ]
            matched_sum += validation["matched_points"]
            if args.diagnose_scale_routing:
                confusion_sum += validation[
                    "route_confusion"
                ]
            if (
                args.expert_only_eval_interval > 0
                and (epoch + 1)
                % args.expert_only_eval_interval
                == 0
            ):
                expert_only_by_dataset[name] = (
                    tm.evaluate_expert_only_mae(
                        model,
                        val_loader,
                        device,
                        temperature=temperature,
                    )
                )

        dataset_count = max(len(per_dataset), 1)
        full3_norm_macro = sum(
            per_dataset[name]["full3"]
            / val_mean_gt_counts[name]
            for name in per_dataset
        ) / dataset_count
        top2_norm_macro = sum(
            per_dataset[name]["top2"]
            / val_mean_gt_counts[name]
            for name in per_dataset
        ) / dataset_count
        top1_norm_macro = sum(
            per_dataset[name]["top1"]
            / val_mean_gt_counts[name]
            for name in per_dataset
        ) / dataset_count
        full3_weighted_norm_mae = weighted_normalized_mae(
            per_dataset,
            val_mean_gt_counts,
            val_image_counts,
            "full3",
        )
        top2_weighted_norm_mae = weighted_normalized_mae(
            per_dataset,
            val_mean_gt_counts,
            val_image_counts,
            "top2",
        )
        top1_weighted_norm_mae = weighted_normalized_mae(
            per_dataset,
            val_mean_gt_counts,
            val_image_counts,
            "top1",
        )
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
        val_gate_entropy = (
            val_gate_entropy_sum
            / max(val_gate_points, 1)
        ).item()
        val_gate_margin = (
            val_gate_margin_sum
            / max(val_gate_points, 1)
        ).item()
        matched_top1_usage = (
            matched_top1_sum
            / max(matched_gate_points, 1)
        )
        matched_top1_usage_string = (
            f"E0:{matched_top1_usage[0] * 100:.1f}% "
            f"E1:{matched_top1_usage[1] * 100:.1f}% "
            f"E2:{matched_top1_usage[2] * 100:.1f}%"
        )
        matched_top1_usage_checkpoint = (
            matched_top1_usage.detach().cpu()
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
        full3_top1_usage_pct = (
            full3_top1_usage_sum.float()
            / max(int(full3_top1_usage_sum.sum()), 1)
            * 100
        )
        per_dataset_str = " | ".join(
            f"{name}: full3={value['full3']:.3f}/"
            f"{value['full3'] / val_mean_gt_counts[name]:.4f} "
            f"top2={value['top2']:.3f}/"
            f"{value['top2'] / val_mean_gt_counts[name]:.4f} "
            f"top1={value['top1'] / val_mean_gt_counts[name]:.4f}"
            for name, value in per_dataset.items()
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
            f"MAE_full3={full3_weighted_norm_mae:.6f} "
            f"MAE_top2={top2_weighted_norm_mae:.6f} "
            f"MAE_top1={top1_weighted_norm_mae:.6f} "
            f"top2_full3_gap={top2_full3_gap:.6f} "
            f"top1_top2_gap={top1_top2_gap:.6f}"
        )
        logging.info("  分数据集: %s", per_dataset_str)
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
        if expert_only_by_dataset:
            expert_only_string = " | ".join(
                f"{name}: E0={values['E0']:.3f} "
                f"E1={values['E1']:.3f} "
                f"E2={values['E2']:.3f}"
                for name, values
                in expert_only_by_dataset.items()
            )
            logging.info(
                "  expert-only MAE: %s",
                expert_only_string,
            )

        if args.diagnose_scale_routing and matched_sum > 0:
            confusion_pct = (
                100.0
                * confusion_sum.float()
                / confusion_sum.sum(
                    dim=1, keepdim=True
                ).clamp_min(1)
            )
            logging.info(
                "  scale-routing confusion "
                "(diagnostic only; GT class -> predicted expert): "
                "E0=[%.1f, %.1f, %.1f]%% n=%d | "
                "E1=[%.1f, %.1f, %.1f]%% n=%d | "
                "E2=[%.1f, %.1f, %.1f]%% n=%d",
                confusion_pct[0, 0],
                confusion_pct[0, 1],
                confusion_pct[0, 2],
                int(confusion_sum[0].sum()),
                confusion_pct[1, 0],
                confusion_pct[1, 1],
                confusion_pct[1, 2],
                int(confusion_sum[1].sum()),
                confusion_pct[2, 0],
                confusion_pct[2, 1],
                confusion_pct[2, 2],
                int(confusion_sum[2].sum()),
            )

        # ---- TensorBoard 写入 ----
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/cls", avg_loss_items["cls"], epoch)
        writer.add_scalar("loss/point", avg_loss_items["point"], epoch)
        writer.add_scalar("loss/count", avg_loss_items["count"], epoch)

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
        for name, values in per_dataset.items():
            normalizer = val_mean_gt_counts[name]
            for mode in ("full3", "top2"):
                writer.add_scalar(
                    f"mae/{name}/{mode}_raw",
                    values[mode],
                    epoch,
                )
                writer.add_scalar(
                    f"mae/{name}/{mode}_norm",
                    values[mode] / normalizer,
                    epoch,
                )
            writer.add_scalar(
                f"mae/{name}/top1_norm",
                values["top1"] / normalizer,
                epoch,
            )

        writer.add_scalar("schedule/router_temperature", temperature, epoch)
        writer.add_scalar("routing/full3_entropy", val_gate_entropy, epoch)
        writer.add_scalar("routing/full3_margin", val_gate_margin, epoch)
        for expert_index in range(3):
            writer.add_scalar(
                f"routing/full3_top1_usage_E{expert_index}_pct",
                float(full3_top1_usage_pct[expert_index]),
                epoch,
            )
        for name, values in expert_only_by_dataset.items():
            for expert_index in range(3):
                writer.add_scalar(
                    f"mae/{name}/expert_only_E{expert_index}",
                    values[f"E{expert_index}"],
                    epoch,
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

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mae": best_selection_score,
            "best_selection_score": best_selection_score,
            "selection_metric": (
                "top2 weighted normalized MAE"
            ),
            "full3_norm_macro": full3_norm_macro,
            "top2_norm_macro": top2_norm_macro,
            "top1_norm_macro": top1_norm_macro,
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
            "per_dataset": per_dataset,
            "expert_only_mae": expert_only_by_dataset,
            "val_image_counts": val_image_counts,
            "val_mean_gt_counts": val_mean_gt_counts,
            "full3_matched_probability_mean": (
                matched_probability_mean.detach().cpu()
            ),
            "matched_top1_usage": (
                matched_top1_usage_checkpoint
            ),
            "full3_top1_usage": (
                full3_top1_usage_sum.detach().cpu()
            ),
            "full3_gate_entropy": val_gate_entropy,
            "full3_gate_margin": val_gate_margin,
            "train_sampled_usage": (
                train_sampled_usage_checkpoint
            ),
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
                confusion_sum.detach().cpu()
                if args.diagnose_scale_routing
                else None
            ),
            "args": vars(args),
            "config": tm.build_checkpoint_config(
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

    writer.close()
    logging.info("训练结束。")



def build_parser():
    parser = tm.build_parser()
    parser.description = (
        "在全部数据集上联合训练（batch 混合 + 逐数据集验证）；"
        "candidate Drop-1 + Soft Top-2"
    )
    parser.add_argument(
        "--dataset", type=str, action="append", default=[],
        metavar="NAME=ROOT:TRAIN[:TRAIN2...]:EVAL",
        help="可多次指定；默认覆盖全部 4 个数据集（见模块 docstring）",
    )
    parser.add_argument(
        "--allow-test-as-eval",
        action="store_true",
        help="显式允许 test split 参与训练期评估（默认禁止）",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_all(parse_args())
