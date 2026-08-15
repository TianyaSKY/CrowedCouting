"""验证集批量推理脚本。

对 datasets/shanghaitech_AB/images/val 全量执行 MoE 点检测推理，
预处理与训练验证完全一致（letterbox 保持纵横比 + 居中填充，
见 scripts/data/point_dataset._letterbox），预测点逆映射回原图后与
GT 点一起绘制，输出到独立目录，同时保存逐图计数 CSV 与 MAE/RMSE 汇总。

用法:
    python -m scripts.visualization.predict_moe_batch \
        --data-root datasets/shanghaitech_AB \
        --checkpoint runs/moe_point/best.pt \
        --out-dir runs/moe_point/val_pred

输出目录结构:
    <out-dir>/images/<name>_pred.jpg   可视化（黄=GT，红/绿/蓝=三专家预测）
    <out-dir>/predictions.csv          逐图 gt/pred 计数
    <out-dir>/summary.json             MAE/RMSE 与专家使用统计
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.yolo11_moe_point import YOLO11MoEPoint
from scripts.visualization.predict_moe import (
    EXPERT_COLORS,
    load_model,
)

# GT 点: 黄色空心圆
GT_COLOR = (0, 255, 255)


def letterbox(
    image_bgr: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, float, int, int]:
    """与 scripts/data/point_dataset._letterbox 保持一致的预处理。

    返回 (填充后的图像, 缩放系数, pad_x, pad_y)，用于把预测点
    从 crop 坐标映射回原图坐标。
    """
    height, width = image_bgr.shape[:2]

    scale = min(
        crop_size / max(width, 1),
        crop_size / max(height, 1),
    )
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (crop_size - new_width) // 2
    pad_y = (crop_size - new_height) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        crop_size - new_height - pad_y,
        pad_x,
        crop_size - new_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, pad_x, pad_y


def load_gt_points(
    point_path: str,
    width: int,
    height: int,
) -> np.ndarray:
    """读取点标注（每行归一化 nx ny），返回原图像素坐标 [N, 2]。"""
    points = []
    if os.path.exists(point_path):
        with open(point_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    nx, ny = float(parts[0]), float(parts[1])
                    points.append([nx * width, ny * height])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def draw_result(
    image_bgr: np.ndarray,
    pred_points: np.ndarray,
    pred_routes: np.ndarray,
    gt_points: np.ndarray,
) -> np.ndarray:
    """绘制 GT（黄色空心圆）与预测点（专家色实心圆）。"""
    result = image_bgr.copy()

    for point in gt_points:
        x, y = int(round(float(point[0]))), int(
            round(float(point[1]))
        )
        cv2.circle(
            result, (x, y), radius=4, color=GT_COLOR, thickness=2
        )

    for point, route in zip(pred_points, pred_routes):
        x, y = int(round(float(point[0]))), int(
            round(float(point[1]))
        )
        color = EXPERT_COLORS[int(route) % len(EXPERT_COLORS)]
        cv2.circle(
            result, (x, y), radius=3, color=color, thickness=-1
        )

    cv2.putText(
        result,
        f"GT: {len(gt_points)}  Pred: {len(pred_points)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return result


def predict_batch(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = load_model(args.weights, args.checkpoint, device)

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not image_paths:
        raise FileNotFoundError(
            f"未在 {image_dir} 中找到任何 jpg 图片"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    image_out_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "predictions.csv")

    total_abs_error = 0.0
    total_squared_error = 0.0
    expert_counts = np.zeros(3, dtype=np.int64)
    num_images = 0

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["filename", "gt_count", "pred_count", "abs_err"]
        )

        for image_path in tqdm(
            image_paths, desc="批量推理", leave=False
        ):
            base_name = os.path.splitext(
                os.path.basename(image_path)
            )[0]

            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"警告: 无法读取 {image_path}，跳过")
                continue

            height, width = image_bgr.shape[:2]

            padded, scale, pad_x, pad_y = letterbox(
                image_bgr, args.imgsz
            )

            image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(
                    image_rgb.astype(np.float32) / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                predictions = model(
                    tensor,
                    temperature=0.5,
                    hard_route=True,
                )

            scores = predictions["logits"].sigmoid()[0]
            crop_points = predictions["points"][0]
            routes = predictions["gates"].argmax(dim=-1)[0]

            keep = scores > args.conf
            crop_points = crop_points[keep].cpu().numpy()
            routes = routes[keep].cpu().numpy()

            # crop 坐标 -> 原图坐标（逆 letterbox）
            pred_points = (
                crop_points
                - np.array([pad_x, pad_y], dtype=np.float32)
            ) / scale
            pred_points[:, 0] = np.clip(
                pred_points[:, 0], 0, width - 1
            )
            pred_points[:, 1] = np.clip(
                pred_points[:, 1], 0, height - 1
            )

            point_path = os.path.join(
                points_dir, base_name + ".txt"
            )
            gt_points = load_gt_points(
                point_path, width, height
            )

            gt_count = gt_points.shape[0]
            pred_count = len(pred_points)
            abs_err = abs(pred_count - gt_count)

            total_abs_error += abs_err
            total_squared_error += abs_err * abs_err
            num_images += 1

            for route in range(3):
                expert_counts[route] += int(
                    (routes == route).sum()
                )

            result = draw_result(
                image_bgr, pred_points, routes, gt_points
            )
            out_image_path = os.path.join(
                image_out_dir, base_name + "_pred.jpg"
            )
            cv2.imwrite(out_image_path, result)

            writer.writerow(
                [base_name, gt_count, pred_count, abs_err]
            )

    if num_images == 0:
        raise RuntimeError("没有成功处理的图像")

    mae = total_abs_error / num_images
    rmse = np.sqrt(total_squared_error / num_images)

    summary = {
        "num_images": num_images,
        "mae": float(mae),
        "rmse": float(rmse),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "checkpoint": args.checkpoint,
        "expert_usage": {
            f"expert{i}": int(expert_counts[i])
            for i in range(3)
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"共处理 {num_images} 张验证图像")
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}")
    print(
        "专家使用: "
        + ", ".join(
            f"E{i}={int(expert_counts[i])}" for i in range(3)
        )
    )
    print(f"可视化结果: {image_out_dir}")
    print(f"逐图计数: {csv_path}")
    print(f"汇总: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证集批量推理：letterbox 预处理 + 可视化 + 计数指标"
    )
    parser.add_argument(
        "--data-root", type=str,
        default="datasets/shanghaitech_AB",
        help="数据集根目录（含 images/ 与 points/）"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="批量推理的分割（val/test）"
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11n.pt",
        help="YOLO11 预训练权重（用于构建 Backbone+Neck）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="runs/moe_point/best.pt",
        help="训练好的 MoE 模型权重"
    )
    parser.add_argument(
        "--imgsz", type=int, default=384,
        help="推理输入尺寸（与训练 crop_size 一致）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="runs/moe_point/val_pred",
        help="结果输出目录（可视化/CSV/汇总均保存于此）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
