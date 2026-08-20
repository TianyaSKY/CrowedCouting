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
import logging
import os

# 与训练一致：避免 Blackwell(RTX50) 上缓存分配器碎片化
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from scripts.data.point_dataset import letterbox_image
from scripts.visualization.predict_moe import (
    EXPERT_COLORS,
    load_model,
    resolve_inference_settings,
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

# GT 点: 黄色空心圆
GT_COLOR = (0, 255, 255)


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

    # 在图像上方增加独立标题横条，不遮挡图像内容
    header_height = 42
    header = np.full((header_height, result.shape[1], 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(
        header,
        f"GT Count: {len(gt_points)} | Visible Pred: {len(pred_points)}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, result])


def predict_batch(args):
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "predict.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"使用设备: {device}")

    model = load_model(args.weights, args.checkpoint, device)
    imgsz, temperature = resolve_inference_settings(
        args.checkpoint,
        imgsz=args.imgsz,
        temperature=args.temperature,
    )
    logging.info(
        "推理设置: imgsz=%d temperature=%.4f",
        imgsz,
        temperature,
    )

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    if not image_paths:
        raise FileNotFoundError(
            f"未在 {image_dir} 中找到任何 jpg 图片"
        )

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
                logging.warning(
                    f"无法读取 {image_path}，跳过"
                )
                continue

            height, width = image_bgr.shape[:2]

            padded, scale, pad_x, pad_y = letterbox_image(
                image_bgr, imgsz
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
                    temperature=temperature,
                    routing_mode="top2",
                )

            scores = predictions["logits"].sigmoid()[0]
            crop_points = predictions["points"][0]
            routes = predictions["gates"].argmax(dim=-1)[0]

            # soft 口径与训练验证一致：全部候选 sigmoid 求和；
            # 可视化仍用阈值点，否则密集图会画满整幅图
            if args.count_mode == "soft":
                pred_count = float(scores.sum().item())
                keep = scores > args.conf
            else:
                keep = scores > args.conf
                pred_count = int(keep.sum().item())

            if args.heatmap:
                # 概率场 -> 密度热力图：每网格单元取 K 个参考点的最大置信度
                side = imgsz // model.point_head.output_stride
                num_refs = model.point_head.num_references
                cell_conf = (
                    scores.view(side, side, num_refs)
                    .max(dim=-1)
                    .values.cpu()
                    .numpy()
                    .astype(np.float32)
                )
                heat = cv2.resize(
                    cell_conf,
                    (imgsz, imgsz),
                    interpolation=cv2.INTER_LINEAR,
                )
                # 去掉 letterbox padding，映射回原图分辨率
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                heat = heat[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
                heat = cv2.resize(
                    heat,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                heat_vis = cv2.applyColorMap(
                    (np.clip(heat, 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_JET,
                )
                overlay = cv2.addWeighted(
                    image_bgr,
                    1.0 - args.heat_alpha,
                    heat_vis,
                    args.heat_alpha,
                    0,
                )
                heat_dir = os.path.join(args.out_dir, "heatmaps")
                os.makedirs(heat_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(
                        heat_dir, base_name + "_heat.jpg"
                    ),
                    overlay,
                )

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
        "count_mode": args.count_mode,
        "heatmap": args.heatmap,
        "imgsz": imgsz,
        "temperature": temperature,
        "checkpoint": args.checkpoint,
        "expert_usage": {
            f"expert{i}": int(expert_counts[i])
            for i in range(3)
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"共处理 {num_images} 张验证图像")
    logging.info(f"MAE={mae:.3f}  RMSE={rmse:.3f}")
    logging.info(
        "专家使用: "
        + ", ".join(
            f"E{i}={int(expert_counts[i])}" for i in range(3)
        )
    )
    logging.info(f"可视化结果: {image_out_dir}")
    logging.info(f"逐图计数: {csv_path}")
    logging.info(f"汇总: {summary_path}")


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
        "--weights", type=str, default="yolo11m.pt",
        help="YOLO11 预训练权重（用于构建 Backbone+Neck）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="runs/moe_point/best_top2.pt",
        help="训练好的 D2 best_top2.pt checkpoint"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="推理输入尺寸（默认读取 checkpoint 的 crop_size）"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Router 推理温度（默认读取 checkpoint 的 config/temperature；"
        "可通过本参数覆盖）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值（thresh 计数与可视化点筛选；soft 计数不受影响）"
    )
    parser.add_argument(
        "--count-mode", type=str,
        choices=["soft", "thresh"], default="thresh",
        help="计数口径: soft=所有候选 sigmoid 之和(与训练验证一致), "
        "thresh=置信度大于 conf 的点数"
    )
    parser.add_argument(
        "--heatmap", action="store_true",
        help="额外输出概率场热力图叠加到原图 (out_dir/heatmaps/)"
    )
    parser.add_argument(
        "--heat-alpha", type=float, default=0.45,
        help="热力图叠加透明度 (0-1)"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="runs/moe_point/val_pred",
        help="结果输出目录（可视化/CSV/汇总均保存于此）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
