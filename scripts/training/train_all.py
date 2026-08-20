"""Joint native_multiscale training with per-dataset validation.

Training concatenates the requested train splits.  Validation runs separately
for each dataset and selects ``best_native.pt`` by weighted normalized MAE.
"""

import argparse
import glob
import math
import os
import shutil
import time

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


NATIVE_ARCHITECTURE = tm.NATIVE_ARCHITECTURE
DEFAULT_DATASETS = [
    "shanghaitech=datasets/shanghaitech_AB:train:val",
    "jhu=datasets/jhu_crowd:train:val",
    "qnrf=datasets/ucf_qnrf:train:val",
    "cc50=datasets/ucf_cc50:fold0_train:fold0_val",
]


def parse_dataset_spec(spec: str) -> tuple[str, str, list[str], str]:
    name, _, rest = spec.partition("=")
    root, _, splits = rest.partition(":")
    train_part, _, eval_split = splits.partition(":")
    train_splits = [split for split in train_part.split("+") if split]
    if not (name and root and train_splits and eval_split):
        raise SystemExit(
            "--dataset 格式应为 name=root:train[:train2...]:eval，"
            f"收到: {spec}"
        )
    return name, root, train_splits, eval_split


def assert_disjoint_splits(
    root: str,
    train_splits: list[str],
    eval_split: str,
) -> None:
    train_names = set()
    for split in train_splits:
        train_names.update(
            os.path.basename(path)
            for path in glob.glob(
                os.path.join(root, "images", split, "*.jpg")
            )
        )
    eval_names = {
        os.path.basename(path)
        for path in glob.glob(
            os.path.join(root, "images", eval_split, "*.jpg")
        )
    }
    overlap = train_names & eval_names
    if overlap:
        raise RuntimeError(
            f"检测到 train/eval 数据泄漏: {len(overlap)} 张重复，"
            f"例如 {sorted(overlap)[:10]}"
        )


def reject_test_as_validation(eval_split: str) -> None:
    if "test" in eval_split.lower():
        raise ValueError(
            f"训练期间禁止使用 test split 选模: {eval_split}"
        )


def reject_test_as_training(train_splits: list[str]) -> None:
    invalid = [split for split in train_splits if "test" in split.lower()]
    if invalid:
        raise ValueError(f"训练期间禁止使用 test split: {invalid}")


def reject_multiple_outer_fold_train_splits(
    train_splits: list[str],
) -> None:
    fold_train_splits = [
        split
        for split in train_splits
        if split.lower().startswith("fold")
        and split.lower().endswith("_train")
    ]
    if len(fold_train_splits) > 1:
        raise ValueError(
            "CC50 每次只能训练一个 outer fold，禁止拼接: "
            f"{fold_train_splits}"
        )


def mean_gt_count(dataset: PointDataset) -> float:
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
    mean_count = total_points / max(len(dataset), 1)
    if mean_count <= 0:
        raise ValueError(
            f"验证集 {dataset.image_dir} 的平均 GT 人数为 0"
        )
    return mean_count


def weighted_normalized_mae(
    per_dataset: dict[str, dict[str, float]],
    val_mean_gt_counts: dict[str, float],
    val_image_counts: dict[str, int],
) -> float:
    total_images = sum(val_image_counts[name] for name in per_dataset)
    if total_images <= 0:
        return float("inf")
    return sum(
        val_image_counts[name]
        * per_dataset[name]["native"]
        / val_mean_gt_counts[name]
        for name in per_dataset
    ) / total_images


def _load_native_resume(
    model,
    optimizer,
    args,
    native_references: tuple[int, int, int],
):
    if not args.resume:
        return optimizer, 0, float("inf")

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
    if tuple(config.get("native_references", ())) != native_references:
        raise ValueError(
            "resume checkpoint 的 native_references 与当前配置不一致"
        )
    model.load_state_dict(checkpoint["model"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except (ValueError, RuntimeError) as error:
        logging.warning("优化器状态不兼容，使用全新优化器: %s", error)
        optimizer = tm.build_optimizer(model, args)
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_score = float(
        checkpoint.get("best_selection_score", float("inf"))
    )
    return optimizer, start_epoch, best_score


def train_all(args: argparse.Namespace) -> None:
    if args.native_warmup_epochs < 0:
        raise ValueError("--native-warmup-epochs 不能为负数")
    native_references = tm.parse_native_references(args.native_references)

    if args.save_dir is None:
        if args.resume:
            args.save_dir = os.path.dirname(os.path.abspath(args.resume))
        else:
            args.save_dir = tm.timestamped_save_dir(
                "runs/native_multiscale_all"
            )
    os.makedirs(args.save_dir, exist_ok=True)
    tm.setup_logging(os.path.join(args.save_dir, "train.log"))
    logging.info("输出目录: %s", args.save_dir)

    writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, "tensorboard")
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)
    tm.validate_cuda_device(device)

    raw_specs = args.dataset or DEFAULT_DATASETS
    specs = [parse_dataset_spec(spec) for spec in raw_specs]
    for name, root, train_splits, eval_split in specs:
        del name
        reject_test_as_training(train_splits)
        reject_multiple_outer_fold_train_splits(train_splits)
        if not args.allow_test_as_eval:
            reject_test_as_validation(eval_split)
        assert_disjoint_splits(root, train_splits, eval_split)

    logging.info("联合数据集:")
    for name, root, train_splits, eval_split in specs:
        logging.info(
            "  %s: train=%s (%s) eval=%s",
            name,
            train_splits,
            root,
            eval_split,
        )

    if not args.resume:
        old_best = os.path.join(args.save_dir, "best_native.pt")
        if os.path.exists(old_best):
            backup = os.path.join(args.save_dir, "best_native_prev.pt")
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info("旧 best_native.pt 已备份到 %s", backup)

    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        native_references=native_references,
    ).to(device)
    criterion = PointMoELoss(match_top_k=args.match_top_k)

    train_datasets = []
    train_dataset_sizes: dict[str, int] = {}
    for name, root, train_splits, _ in specs:
        for split in train_splits:
            dataset = PointDataset(
                root,
                split=split,
                crop_size=args.crop_size,
                augment=True,
            )
            train_datasets.append(dataset)
            train_dataset_sizes[name] = (
                train_dataset_sizes.get(name, 0) + len(dataset)
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
    for name, root, _, eval_split in specs:
        val_dataset = PointDataset(
            root,
            split=eval_split,
            crop_size=args.crop_size,
            augment=False,
        )
        val_mean_gt_counts[name] = mean_gt_count(val_dataset)
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

    val_image_interval, val_image_count, val_image_conf = (
        tm.validation_image_options(args)
    )
    logging.info(
        "训练样本总数: %d（%d 个数据集 split 拼接）",
        len(train_dataset),
        len(train_datasets),
    )
    logging.info(
        "训练按图片自然采样: %s",
        ", ".join(
            f"{name}={count} ({count / max(len(train_dataset), 1):.1%})"
            for name, count in train_dataset_sizes.items()
        ),
    )

    if args.freeze_epochs > 0:
        for param in model.yolo.parameters():
            param.requires_grad = False
        logging.info(
            "前 %d 个 epoch 冻结 YOLO Backbone+Neck",
            args.freeze_epochs,
        )

    optimizer = tm.build_optimizer(model, args)
    optimizer, start_epoch, best_selection_score = _load_native_resume(
        model,
        optimizer,
        args,
        native_references,
    )

    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.perf_counter()
        matching_mode = tm.native_matching_mode(epoch, args)
        matching_stage = tm.native_matching_stage(epoch, args)
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

        per_dataset: dict[str, dict[str, float]] = {}
        validation_by_dataset: dict[str, dict[str, object]] = {}
        expert_only_by_dataset = {}
        val_winner_hist = torch.zeros(
            3, dtype=torch.int64, device=device
        )
        val_positive_count = torch.zeros(3, device=device)
        val_distance_sum = torch.zeros(3, device=device)
        val_confidence_sum = torch.zeros(3, device=device)
        val_matched_count = 0

        for name, val_loader in val_loaders.items():
            validation = tm.evaluate_native_count_mae(
                model,
                val_loader,
                device,
                criterion=criterion,
                max_visual_samples=val_image_count if collect_visuals else 0,
            )
            validation_by_dataset[name] = validation
            per_dataset[name] = {"native": validation["mae"]}
            val_winner_hist += validation["winner_hist"].to(device)
            val_positive_count += validation["positive_count"].to(device)
            val_distance_sum += validation["matched_distance_sum"].to(device)
            val_confidence_sum += validation[
                "matched_confidence_sum"
            ].to(device)
            val_matched_count += validation["matched_count"]

            if collect_visuals and validation["validation_samples"]:
                tm.log_validation_images(
                    writer,
                    f"val_images/{name}",
                    validation["validation_samples"],
                    epoch,
                    conf_threshold=val_image_conf,
                )
            if (
                args.expert_only_eval_interval > 0
                and (epoch + 1) % args.expert_only_eval_interval == 0
            ):
                expert_only_by_dataset[name] = tm.evaluate_expert_only_mae(
                    model,
                    val_loader,
                    device,
                )

        dataset_count = max(len(per_dataset), 1)
        native_macro_norm = sum(
            per_dataset[name]["native"] / val_mean_gt_counts[name]
            for name in per_dataset
        ) / dataset_count
        native_weighted_norm = weighted_normalized_mae(
            per_dataset,
            val_mean_gt_counts,
            val_image_counts,
        )
        total_val_images = max(sum(val_image_counts.values()), 1)
        native_rmse = (
            sum(
                val_image_counts[name]
                * validation_by_dataset[name]["rmse"] ** 2
                for name in validation_by_dataset
            )
            / total_val_images
        ) ** 0.5
        native_bias = sum(
            val_image_counts[name]
            * validation_by_dataset[name]["bias"]
            for name in validation_by_dataset
        ) / total_val_images
        native_loss = {
            loss_name: sum(
                val_image_counts[name]
                * validation_by_dataset[name]["loss"][loss_name]
                for name in validation_by_dataset
            )
            / total_val_images
            for loss_name in ("total", "cls", "point", "count")
        }
        val_distance_mean = val_distance_sum / val_positive_count.clamp_min(1)
        val_confidence_mean = val_confidence_sum / val_positive_count.clamp_min(1)
        train_distance_mean = train_distance_sum / train_positive_sum.clamp_min(1)
        train_confidence_mean = train_confidence_sum / train_positive_sum.clamp_min(1)
        val_winner_pct = val_winner_hist.float() / max(int(val_winner_hist.sum()), 1) * 100
        train_winner_pct = train_winner_sum.float() / max(int(train_winner_sum.sum()), 1) * 100

        logging.info(
            "[Epoch %d/%d] loss=%.4f cls=%.4f point=%.4f count=%.4f stage=%s MAE_native=%.6f RMSE=%.3f bias=%.3f",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_loss_items["cls"],
            avg_loss_items["point"],
            avg_loss_items["count"],
            matching_stage,
            native_weighted_norm,
            native_rmse,
            native_bias,
        )
        logging.info(
            "  分数据集: %s",
            " | ".join(
                f"{name}: {values['native']:.3f}/{values['native'] / val_mean_gt_counts[name]:.4f}"
                for name, values in per_dataset.items()
            ),
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
        for name, validation in validation_by_dataset.items():
            dataset_positive = validation["positive_count"]
            dataset_winner = validation["winner_hist"]
            dataset_distance = validation["matched_distance_sum"].to(device) / dataset_positive.to(device).clamp_min(1)
            dataset_confidence = validation["matched_confidence_sum"].to(device) / dataset_positive.to(device).clamp_min(1)
            dataset_winner_pct = dataset_winner.float() / max(int(dataset_winner.sum()), 1) * 100
            logging.info(
                "  %s winner: E0=%.1f%% E1=%.1f%% E2=%.1f%% | distance: %.2f/%.2f/%.2f | confidence: %.3f/%.3f/%.3f",
                name,
                dataset_winner_pct[0],
                dataset_winner_pct[1],
                dataset_winner_pct[2],
                dataset_distance[0],
                dataset_distance[1],
                dataset_distance[2],
                dataset_confidence[0],
                dataset_confidence[1],
                dataset_confidence[2],
            )
        if expert_only_by_dataset:
            logging.info("  expert-only MAE: %s", expert_only_by_dataset)

        for name, value in (
            ("total", avg_loss),
            ("cls", avg_loss_items["cls"]),
            ("point", avg_loss_items["point"]),
            ("count", avg_loss_items["count"]),
        ):
            writer.add_scalar(f"loss/{name}", value, epoch)
            writer.add_scalar(f"train/loss_{name}", value, epoch)
        for name, value in native_loss.items():
            writer.add_scalar(f"val/loss_native_{name}", value, epoch)
        writer.add_scalar("mae/native_weighted_norm", native_weighted_norm, epoch)
        writer.add_scalar("mae/native_macro_norm", native_macro_norm, epoch)
        writer.add_scalar("val/rmse_native", native_rmse, epoch)
        writer.add_scalar("val/count_bias_native", native_bias, epoch)
        writer.add_scalar("native/matched_count", val_matched_count, epoch)
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
        for name, validation in validation_by_dataset.items():
            dataset_positive = validation["positive_count"]
            dataset_winner = validation["winner_hist"]
            dataset_distance = validation["matched_distance_sum"].to(device) / dataset_positive.to(device).clamp_min(1)
            dataset_confidence = validation["matched_confidence_sum"].to(device) / dataset_positive.to(device).clamp_min(1)
            dataset_winner_pct = dataset_winner.float() / max(int(dataset_winner.sum()), 1) * 100
            for expert_index in range(3):
                writer.add_scalar(f"native/{name}/winner_E{expert_index}_pct", float(dataset_winner_pct[expert_index]), epoch)
                writer.add_scalar(f"native/{name}/matched_distance_E{expert_index}", float(dataset_distance[expert_index]), epoch)
                writer.add_scalar(f"native/{name}/matched_confidence_E{expert_index}", float(dataset_confidence[expert_index]), epoch)
                writer.add_scalar(f"native/{name}/positive_count_E{expert_index}", float(dataset_positive[expert_index]), epoch)

        for name, values in expert_only_by_dataset.items():
            for expert_index in range(3):
                writer.add_scalar(
                    f"mae/{name}/expert_only_E{expert_index}",
                    values[f"E{expert_index}"],
                    epoch,
                )

        improved = (
            matching_mode == "competitive"
            and native_weighted_norm < best_selection_score
        )
        if improved:
            best_selection_score = native_weighted_norm
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
            "native_weighted_norm_mae": native_weighted_norm,
            "native_macro_norm_mae": native_macro_norm,
            "native_rmse": native_rmse,
            "native_bias": native_bias,
            "native_loss": native_loss,
            "matching_stage": matching_stage,
            "native_winner_hist": val_winner_hist.detach().cpu(),
            "native_train_winner_hist": train_winner_sum.detach().cpu(),
            "native_positive_count": val_positive_count.detach().cpu(),
            "native_matched_distance_mean": val_distance_mean.detach().cpu(),
            "native_matched_confidence_mean": val_confidence_mean.detach().cpu(),
            "native_matched_count": val_matched_count,
            "per_dataset": per_dataset,
            "expert_only_mae": expert_only_by_dataset,
            "args": vars(args),
            "config": tm.build_checkpoint_config(
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
        tm.log_system_metrics(
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
    parser = tm.build_parser()
    parser.description = (
        "联合训练 native_multiscale P3/P4/P5 experts；"
        "独立 warmup 后进行全局 Hungarian competition"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        default=[],
        metavar="NAME=ROOT:TRAIN[:TRAIN2...]:EVAL",
        help="可多次指定；默认使用四个数据集",
    )
    parser.add_argument(
        "--allow-test-as-eval",
        action="store_true",
        help="允许 test split 参与训练期评估",
    )
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_all(parse_args())
