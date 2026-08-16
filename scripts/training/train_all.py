"""在所有数据集上联合训练（batch 混合）+ 逐数据集验证。

训练数据 = 各数据集 train split 的 ConcatDataset（batch 内随机混合），
验证 = 每个数据集各自跑 soft/hard MAE；Router 毕业判定用各数据集
混淆矩阵之和，best 模型按各数据集 mean-GT-count 归一化后的 MAE
算术平均选取（每个数据集等权）。官方 test 只允许在训练结束后评估。

用法（GPU 机器，从项目根目录）:

    python -m scripts.training.train_all \
        --weights yolo11n.pt \
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


def train_all(args: argparse.Namespace) -> None:
    # 未显式指定输出目录时自动加时间戳；--resume 沿用原 run 目录
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
    if args.resume:
        logging.warning(
            "resume 仅用于调试；正式修复实验必须从 yolo11n.pt 新开 run，"
            "避免继承旧 checkpoint 的 test 泄漏"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    specs = (
        [parse_dataset_spec(s) for s in args.dataset]
        if args.dataset
        else [parse_dataset_spec(s) for s in DEFAULT_DATASETS]
    )
    allow_test_as_eval = getattr(args, "allow_test_as_eval", False)
    for name, root, train_splits, eval_split in specs:
        reject_test_as_training(train_splits)
        reject_multiple_outer_fold_train_splits(train_splits)
        if not allow_test_as_eval:
            reject_test_as_validation(eval_split)
        assert_disjoint_splits(root, train_splits, eval_split)

    logging.info("联合数据集:")
    for name, root, train_splits, eval_split in specs:
        logging.info(
            "  %s: train=%s (%s) eval=%s", name, train_splits, root, eval_split
        )
        if allow_test_as_eval and "test" in eval_split.lower():
            logging.warning(
                "显式允许 test 参与训练期评估: %s (%s)",
                name,
                eval_split,
            )

    graduate_recalls = tuple(
        float(v) for v in args.graduate_recalls.split(",")
    )
    if len(graduate_recalls) != 3:
        raise ValueError("--graduate-recalls 需要 3 个值")

    # 覆盖保护：从头训练进入已有 save-dir 时备份旧 best
    if not args.resume:
        old_best = os.path.join(args.save_dir, "best.pt")
        if os.path.exists(old_best):
            backup = os.path.join(args.save_dir, "best_prev.pt")
            if not os.path.exists(backup):
                shutil.copy2(old_best, backup)
                logging.info(f"旧 best 已备份到 {backup}")

    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        num_references=args.num_references,
    ).to(device)

    criterion = PointMoELoss(
        route_weight=args.route_weight,
        match_top_k=args.match_top_k,
    )

    # 训练集：各数据集 train split 拼接，batch 内随机混合
    train_datasets = []
    for name, root, train_splits, eval_split in specs:
        for split in train_splits:
            train_datasets.append(
                PointDataset(
                    root,
                    split=split,
                    crop_size=args.crop_size,
                    augment=True,
                )
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

    # 验证集：每数据集一个 loader，逐数据集报告 MAE
    val_loaders = {}
    val_mean_gt_counts = {}
    for name, root, train_splits, eval_split in specs:
        val_dataset = PointDataset(
            root,
            split=eval_split,
            crop_size=args.crop_size,
            augment=False,
        )
        val_mean_gt_counts[name] = mean_gt_count(val_dataset)
        val_loaders[name] = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=point_collate_fn,
        )
        logging.info(
            "  %s 验证集平均 GT 人数: %.3f",
            name,
            val_mean_gt_counts[name],
        )
    logging.info(
        "训练样本总数: %d（%d 个数据集 split 拼接）",
        len(train_dataset),
        len(train_datasets),
    )

    # 冻结 / 优化器 / 恢复
    if args.freeze_epochs > 0:
        logging.info(f"前 {args.freeze_epochs} 个 epoch 冻结 YOLO Backbone+Neck")
        for param in model.yolo.parameters():
            param.requires_grad = False

    optimizer = tm.build_optimizer(model, args)

    start_epoch = 0
    best_selection_score = float("inf")
    hard_started = False
    grad_streak = 0
    first_hard_epoch = None

    if args.resume and os.path.exists(args.resume):
        logging.info(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as error:
            logging.warning(
                f"旧版 checkpoint 缺少新参数({error})，缺失部分使用初始化值"
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
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
        hard_started = checkpoint.get("hard_started", False)
        grad_streak = checkpoint.get("grad_streak", 0)
        first_hard_epoch = checkpoint.get("first_hard_epoch")
        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True

    was_hard_route = hard_started

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
            logging.info(f"强制启用硬路由（epoch={args.force_hard_epoch}）")

        if hard_route and not was_hard_route:
            was_hard_route = True
            first_hard_epoch = epoch
            best_selection_score = float("inf")
            logging.info(
                "切换硬路由，best 基准重置为 hard normalized macro MAE"
            )

        temperature = tm.temperature_for_epoch(
            epoch, hard_route, first_hard_epoch, args
        )

        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            logging.info("解冻 YOLO Backbone+Neck（保留优化器状态）")

        model.train()
        total_loss = 0.0
        num_batches = 0
        gate_mean = torch.zeros(3, device=device)
        target_mean = torch.zeros(3, device=device)
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
            gt_points = [p.to(device) for p in batch["points"]]

            predictions = model(
                images,
                temperature=temperature,
                hard_route=hard_route,
                router_grad=router_grad,
            )
            loss, loss_items = criterion(
                predictions, gt_points, image_size=images.shape[-2:]
            )

            gate_mean += predictions["gates"].reshape(-1, 3).mean(dim=0)
            target_mean += loss_items["gate_target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip
            )
            optimizer.step()

            total_loss += loss.item()
            for loss_name in loss_sums:
                loss_sums[loss_name] += loss_items[loss_name].detach()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_items = {
            name: (loss_sums[name] / max(num_batches, 1)).item()
            for name in loss_sums
        }

        # 逐数据集验证
        per_dataset = {}
        confusion_sum = torch.zeros(3, 3, dtype=torch.int64, device=device)
        matched_sum = 0
        for name, val_loader in val_loaders.items():
            soft_mae, hard_mae, _, route_confusion, matched = (
                tm.evaluate_count_mae(
                    model, val_loader, device, args.match_top_k
                )
            )
            per_dataset[name] = (soft_mae, hard_mae)
            confusion_sum += route_confusion
            matched_sum += matched

        soft_raw_macro = sum(
            value[0] for value in per_dataset.values()
        ) / len(per_dataset)
        hard_raw_macro = sum(
            value[1] for value in per_dataset.values()
        ) / len(per_dataset)
        soft_norm_macro = sum(
            per_dataset[name][0] / val_mean_gt_counts[name]
            for name in per_dataset
        ) / len(per_dataset)
        hard_norm_macro = sum(
            per_dataset[name][1] / val_mean_gt_counts[name]
            for name in per_dataset
        ) / len(per_dataset)

        per_dataset_str = " | ".join(
            f"{name}: soft={v[0]:.3f}/{v[0] / val_mean_gt_counts[name]:.4f} "
            f"hard={v[1]:.3f}/{v[1] / val_mean_gt_counts[name]:.4f}"
            for name, v in per_dataset.items()
        )
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
            f"soft_raw_macro={soft_raw_macro:.3f} "
            f"soft_norm_macro={soft_norm_macro:.6f} "
            f"hard_raw_macro={hard_raw_macro:.3f} "
            f"hard_norm_macro={hard_norm_macro:.6f}"
        )
        logging.info("  分数据集: %s", per_dataset_str)

        # Router 毕业（混淆矩阵跨数据集求和）
        if not hard_started and matched_sum > 0:
            recalls, macro_recall = tm.router_recalls(confusion_sum)
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
                        f"Router 毕业（recall=E0:{recalls[0]:.2f} "
                        f"E1:{recalls[1]:.2f} E2:{recalls[2]:.2f} "
                        f"macro={macro_recall:.2f}），下一 epoch 进 hard"
                    )
                else:
                    logging.info(
                        f"Router 达标 streak={grad_streak}/"
                        f"{args.graduate_stable_epochs}"
                    )
            else:
                grad_streak = 0
                logging.info(
                    f"Router 未达标 (recall=E0:{recalls[0]:.2f} "
                    f"E1:{recalls[1]:.2f} E2:{recalls[2]:.2f} "
                    f"macro={macro_recall:.2f})"
                )

        # best 选择：原始 MAE 保留用于日志，checkpoint 使用归一化
        # validation score，避免高人数 CC50 的绝对误差支配所有数据集。
        val_score_for_best = (
            hard_norm_macro if hard_route else soft_norm_macro
        )
        metric_name = (
            "hard normalized macro MAE"
            if hard_route
            else "soft normalized macro MAE"
        )

        if val_score_for_best < best_selection_score:
            best_selection_score = val_score_for_best
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    # 保留 best_mae 作为旧 checkpoint 读取兼容字段；
                    # 现在它表示 normalized selection score。
                    "best_mae": best_selection_score,
                    "best_selection_score": best_selection_score,
                    "selection_metric": metric_name,
                    "hard_started": hard_started,
                    "grad_streak": grad_streak,
                    "first_hard_epoch": first_hard_epoch,
                    "per_dataset": per_dataset,
                    "val_mean_gt_counts": val_mean_gt_counts,
                    "soft_raw_macro": soft_raw_macro,
                    "soft_norm_macro": soft_norm_macro,
                    "hard_raw_macro": hard_raw_macro,
                    "hard_norm_macro": hard_norm_macro,
                    "args": vars(args),
                },
                best_path,
            )
            logging.info(
                f"  -> 新的最佳 {metric_name}: "
                f"{best_selection_score:.6f}"
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_mae": best_selection_score,
                "best_selection_score": best_selection_score,
                "selection_metric": metric_name,
                "hard_started": hard_started,
                "grad_streak": grad_streak,
                "first_hard_epoch": first_hard_epoch,
                "per_dataset": per_dataset,
                "val_mean_gt_counts": val_mean_gt_counts,
                "soft_raw_macro": soft_raw_macro,
                "soft_norm_macro": soft_norm_macro,
                "hard_raw_macro": hard_raw_macro,
                "hard_norm_macro": hard_norm_macro,
                "args": vars(args),
            },
            os.path.join(args.save_dir, "last.pt"),
        )

    logging.info("训练结束。")


def build_parser():
    parser = tm.build_parser()
    parser.description = (
        "在全部数据集上联合训练（batch 混合 + 逐数据集验证）"
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
