import argparse
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.yolo11_moe_point import YOLO11MoEPoint
from models.point_moe_loss import PointMoELoss
from scripts.data.point_dataset import (
    PointDataset,
    point_collate_fn,
)


def build_optimizer(model, args):
    """两个学习率组：YOLO 主干 1e-4，MoE Point Head 1e-3。"""
    head_params = [
        p for p in model.point_head.parameters()
        if p.requires_grad
    ]
    yolo_params = [
        p for p in model.yolo.parameters()
        if p.requires_grad
    ]

    groups = [
        {
            "params": head_params,
            "lr": args.head_lr,
        },
    ]
    if yolo_params:
        groups.append(
            {
                "params": yolo_params,
                "lr": args.backbone_lr,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )


def evaluate_count_mae(model, val_loader, device):
    """验证集人数 MAE（以所有候选点置信度和作为预测人数）。"""
    model.eval()

    total_abs_error = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["img"].to(device)
            gt_points = [
                p.to(device) for p in batch["points"]
            ]

            predictions = model(
                images,
                temperature=0.5,
                hard_route=True,
            )

            scores = predictions["logits"].sigmoid()
            pred_counts = scores.sum(dim=1)

            for i in range(images.shape[0]):
                gt_count = gt_points[i].shape[0]
                total_abs_error += abs(
                    float(pred_counts[i]) - gt_count
                )
                total_images += 1

    model.train()
    return total_abs_error / max(total_images, 1)


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
    criterion = PointMoELoss()

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
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_mae = checkpoint.get("best_mae", float("inf"))

        # 恢复后若已超过冻结期，解冻 YOLO
        if start_epoch >= args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, args)

    # 5. 训练循环
    for epoch in range(start_epoch, args.epochs):
        temperature = max(
            args.min_temperature,
            args.init_temperature
            - args.temperature_decay * epoch / 30.0,
        )
        hard_route = epoch >= args.hard_route_epoch

        # 到达解冻点：解冻 YOLO 并重建优化器（加入新参数组）
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.yolo.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, args)
            print("解冻 YOLO Backbone+Neck，重建优化器")

        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )

            optimizer.step()

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # 6. 验证与保存
        val_mae = evaluate_count_mae(
            model, val_loader, device
        )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} "
            f"cls={float(loss_items['cls']):.4f} "
            f"point={float(loss_items['point']):.4f} "
            f"count={float(loss_items['count']):.4f} "
            f"balance={float(loss_items['balance']):.4f} "
            f"T={temperature:.2f} "
            f"hard_route={hard_route} "
            f"val_MAE={val_mae:.3f}"
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

        if val_mae < best_mae:
            best_mae = val_mae
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
            print(f"  -> 新的最佳 MAE: {best_mae:.3f}，已保存 {best_path}")

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
