import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo11_moe_point import YOLO11MoEPoint
from models.point_moe_loss import PointMoELoss
from scripts.data.point_dataset import (
    PointDataset,
    point_collate_fn,
)


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


def evaluate_count_mae(model, val_loader, device):
    """验证集人数 MAE（以所有候选点置信度和作为预测人数）。

    同时评估软路由与硬路由两种模式：
    - 训练早期为软路由，soft_MAE 与训练一致；
    - hard routing 生效后以 hard_MAE 作为 best 模型选择依据。
    """
    model.eval()

    total_abs_error = {"soft": 0.0, "hard": 0.0}
    total_images = 0
    hard_usage = torch.zeros(3, dtype=torch.int64, device=device)

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
                    temperature=0.5,
                    hard_route=hard_route,
                )

                scores = predictions["logits"].sigmoid()
                pred_counts = scores.sum(dim=1)

                for i, gt_count in enumerate(gt_counts):
                    total_abs_error[mode] += abs(
                        float(pred_counts[i]) - gt_count
                    )

                if hard_route:
                    # 硬路由使用率：所有候选点 argmax 到的专家分布
                    hard_usage += (
                        predictions["gates"]
                        .argmax(dim=-1)
                        .flatten()
                        .bincount(minlength=3)
                    )

            total_images += len(gt_counts)

    model.train()

    soft_mae = total_abs_error["soft"] / max(
        total_images, 1
    )
    hard_mae = total_abs_error["hard"] / max(
        total_images, 1
    )

    return soft_mae, hard_mae, hard_usage


def train_moe(args):
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 模型
    model = YOLO11MoEPoint(
        weights=args.weights,
        hidden_channels=args.hidden_channels,
        num_references=args.num_references,
    ).to(device)

    # 2. 损失
    criterion = PointMoELoss(
        route_weight=args.route_weight,
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
        print(
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
        print(
            f"前 {args.freeze_epochs} 个 epoch 冻结 YOLO Backbone+Neck"
        )
        for param in model.yolo.parameters():
            param.requires_grad = False

    optimizer = build_optimizer(model, args)

    start_epoch = 0
    best_mae = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as error:
            print(
                f"警告: 旧版 checkpoint 缺少新参数({error})，"
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
            print(
                f"警告: 优化器状态不兼容({error})，"
                "使用全新优化器重新开始优化"
            )
            optimizer = build_optimizer(model, args)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_mae = checkpoint.get("best_mae", float("inf"))

        # 恢复到硬路由阶段时，checkpoint 的 best_mae 可能是 soft 口径，
        # 重置基准避免两种口径混用
        if start_epoch >= args.hard_route_epoch:
            best_mae = float("inf")
            print("已进入硬路由阶段，best 基准重置为 hard MAE")

        # 恢复后若已超过冻结期，解冻 YOLO（不重建优化器，保留状态）
        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True

    # 5. 训练循环
    for epoch in range(start_epoch, args.epochs):
        temperature = max(
            args.min_temperature,
            args.init_temperature
            - args.temperature_decay * epoch / 30.0,
        )
        hard_route = epoch >= args.hard_route_epoch

        # 到达解冻点：解冻 YOLO，不重建优化器——
        # 冻结参数在优化器中自动跳过，解冻后自然开始积累 Adam 状态
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            print("解冻 YOLO Backbone+Neck（保留优化器状态，不重建）")

        model.train()
        total_loss = 0.0
        num_batches = 0
        gate_mean = torch.zeros(3, device=device)
        target_mean = torch.zeros(3, device=device)

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
            )

            loss, loss_items = criterion(
                predictions,
                gt_points,
                image_size=images.shape[-2:],
            )

            # 诊断：本 batch 候选点的平均 gate 分布（软阶段=soft 门，
            # 硬阶段=one-hot 使用率），与 GT 目标分布对比
            gate_mean += predictions["gates"].reshape(
                -1, 3
            ).mean(dim=0)
            target_mean += loss_items["gate_target"].to(
                device
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 6. 验证与保存
        soft_mae, hard_mae, hard_usage = evaluate_count_mae(
            model, val_loader, device
        )

        gate_pct = gate_mean / max(num_batches, 1) * 100
        target_pct = target_mean / max(num_batches, 1) * 100
        usage_pct = (
            hard_usage.float()
            / max(int(hard_usage.sum()), 1)
            * 100
        )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} "
            f"cls={float(loss_items['cls']):.4f} "
            f"point={float(loss_items['point']):.4f} "
            f"count={float(loss_items['count']):.4f} "
            f"route={float(loss_items['route']):.4f} "
            f"T={temperature:.2f} "
            f"hard_route={hard_route} "
            f"soft_MAE={soft_mae:.3f} "
            f"hard_MAE={hard_mae:.3f}"
        )
        print(
            f"  gate  =E0:{gate_pct[0]:.1f}% E1:{gate_pct[1]:.1f}% "
            f"E2:{gate_pct[2]:.1f}%"
            f" | target=E0:{target_pct[0]:.1f}% "
            f"E1:{target_pct[1]:.1f}% E2:{target_pct[2]:.1f}%"
            f" | 硬路由使用=E0:{usage_pct[0]:.1f}% "
            f"E1:{usage_pct[1]:.1f}% E2:{usage_pct[2]:.1f}%"
        )

        last_path = os.path.join(args.save_dir, "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "args": vars(args),
            },
            last_path,
        )

        # best 选择：软路由阶段按 soft MAE、硬路由阶段按 hard MAE
        # （最终推理使用硬路由）。切换硬路由时重置基准，
        # 避免 soft 阶段的 best 继续占位。
        if hard_route:
            if epoch == args.hard_route_epoch:
                best_mae = float("inf")
                print("切换硬路由，best 基准重置为 hard MAE")
            val_mae_for_best = hard_mae
            metric_name = "hard MAE"
        else:
            val_mae_for_best = soft_mae
            metric_name = "soft MAE"

        if val_mae_for_best < best_mae:
            best_mae = val_mae_for_best
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mae": best_mae,
                    "args": vars(args),
                },
                best_path,
            )
            print(
                f"  -> 新的最佳 {metric_name}: {best_mae:.3f}，"
                f"已保存 {best_path}"
            )

    print("训练结束。")


def parse_args():
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
        "--crop-size", type=int, default=384,
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
        "--backbone-lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--head-lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4
    )
    parser.add_argument(
        "--init-temperature", type=float, default=2.0
    )
    parser.add_argument(
        "--min-temperature", type=float, default=0.5
    )
    parser.add_argument(
        "--temperature-decay", type=float, default=1.5,
        help="温度随 epoch 衰减量（30 epoch 内衰减完）"
    )
    parser.add_argument(
        "--hard-route-epoch", type=int, default=20,
        help="启用 Top-1 硬路由的起始 epoch"
    )
    parser.add_argument(
        "--route-weight", type=float, default=0.05,
        help="尺度路由监督(CE)损失权重"
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
        "--save-dir", type=str,
        default="runs/moe_point",
        help="权重保存目录"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 恢复训练"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_moe(parse_args())
