"""在所有数据集上联合训练（batch 混合）+ 逐数据集验证。

训练数据 = 各数据集 train split 的 ConcatDataset（batch 内随机混合），
验证 = 每个数据集各自跑 soft/hard MAE；Router 毕业判定用各数据集
混淆矩阵之和，best 模型按各数据集 MAE 的算术平均选取（每个数据集等权）。

用法（GPU 机器，从项目根目录）:

    python -m scripts.training.train_all \
        --weights yolo11n.pt \
        --save-dir runs/moe_point_all \
        --epochs 100

默认数据集（--dataset 可覆盖，可重复）:
    shanghaitech=datasets/shanghaitech_AB:train:val
    jhu=datasets/jhu_crowd:train:val
    qnrf=datasets/ucf_qnrf:train:test
    cc50=datasets/ucf_cc50:fold0_train+fold1_train+fold2_train+fold3_train+fold4_train:fold0_test

--dataset 格式: name=root:train_split[:train_split2...]:eval_split
（train split 用 + 连接多个；CC-50 联合训练默认吃满全部 50 张，
 验证用 fold0_test 留出 10 张。）

超参数与 train_moe 完全一致（共享 argparse），checkpoint 格式兼容
evaluate_datasets / test_each_dataset。先转换数据:

    python -m scripts.data.prepare_all
"""

import argparse
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
    "qnrf=datasets/ucf_qnrf:train:test",
    "cc50=datasets/ucf_cc50:"
    "fold0_train+fold1_train+fold2_train+fold3_train+fold4_train:"
    "fold0_test",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    specs = (
        [parse_dataset_spec(s) for s in args.dataset]
        if args.dataset
        else [parse_dataset_spec(s) for s in DEFAULT_DATASETS]
    )
    logging.info("联合数据集:")
    for name, root, train_splits, eval_split in specs:
        logging.info(
            "  %s: train=%s (%s) eval=%s", name, train_splits, root, eval_split
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
    for name, root, train_splits, eval_split in specs:
        val_dataset = PointDataset(
            root,
            split=eval_split,
            crop_size=args.crop_size,
            augment=False,
        )
        val_loaders[name] = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=point_collate_fn,
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
    best_mae = float("inf")
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
        best_mae = checkpoint.get("best_mae", float("inf"))
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
            best_mae = float("inf")
            logging.info("切换硬路由，best 基准重置为 hard MAE")

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

        soft_overall = sum(v[0] for v in per_dataset.values()) / len(
            per_dataset
        )
        hard_overall = sum(v[1] for v in per_dataset.values()) / len(
            per_dataset
        )

        per_dataset_str = " | ".join(
            f"{name}: soft={v[0]:.3f}/hard={v[1]:.3f}"
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
            f"soft_overall={soft_overall:.3f} "
            f"hard_overall={hard_overall:.3f}"
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

        # best 选择：软阶段按 soft 整体 MAE、硬阶段按 hard 整体 MAE
        val_mae_for_best = hard_overall if hard_route else soft_overall
        metric_name = "hard overall MAE" if hard_route else "soft overall MAE"

        if val_mae_for_best < best_mae:
            best_mae = val_mae_for_best
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mae": best_mae,
                    "hard_started": hard_started,
                    "grad_streak": grad_streak,
                    "first_hard_epoch": first_hard_epoch,
                    "per_dataset": per_dataset,
                    "args": vars(args),
                },
                best_path,
            )
            logging.info(f"  -> 新的最佳 {metric_name}: {best_mae:.3f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "hard_started": hard_started,
                "grad_streak": grad_streak,
                "first_hard_epoch": first_hard_epoch,
                "per_dataset": per_dataset,
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
    return parser


def parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    train_all(parse_args())
