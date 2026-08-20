"""Batch inference and visualization for the native_multiscale head."""

import argparse
import csv
import glob
import json
import logging
import os

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

GT_COLOR = (0, 255, 255)


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


def load_gt_points(point_path: str, width: int, height: int) -> np.ndarray:
    points = []
    if os.path.exists(point_path):
        with open(point_path, encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 2:
                    points.append(
                        [float(parts[0]) * width, float(parts[1]) * height]
                    )
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def draw_result(
    image_bgr: np.ndarray,
    pred_points: np.ndarray,
    pred_sources: np.ndarray,
    gt_points: np.ndarray,
) -> np.ndarray:
    result = image_bgr.copy()
    for point in gt_points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(result, (x, y), radius=4, color=GT_COLOR, thickness=2)
    for point, source in zip(pred_points, pred_sources):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(
            result,
            (x, y),
            radius=3,
            color=EXPERT_COLORS[int(source) % len(EXPERT_COLORS)],
            thickness=-1,
        )
    header = np.full(
        (42, result.shape[1], 3),
        (24, 24, 24),
        dtype=np.uint8,
    )
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


def native_heatmap(scores, model, imgsz: int) -> np.ndarray:
    heat = np.zeros((imgsz, imgsz), dtype=np.float32)
    offset = 0
    for stride, references in zip(
        model.point_head.output_strides,
        model.point_head.references_per_expert,
    ):
        height_cells = imgsz // int(stride)
        width_cells = imgsz // int(stride)
        count = height_cells * width_cells * references
        level_scores = scores[offset:offset + count]
        offset += count
        if level_scores.numel() != count:
            continue
        level_conf = (
            level_scores.reshape(height_cells, width_cells, references)
            .max(dim=-1)
            .values.cpu()
            .numpy()
        )
        heat = np.maximum(
            heat,
            cv2.resize(
                level_conf,
                (imgsz, imgsz),
                interpolation=cv2.INTER_LINEAR,
            ),
        )
    return heat


def predict_batch(args):
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "predict.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, args.checkpoint, device)
    imgsz = resolve_inference_settings(args.checkpoint, imgsz=args.imgsz)
    logging.info("Native 推理设置: imgsz=%d", imgsz)

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到任何 jpg 图片")

    image_out_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    total_abs_error = 0.0
    total_squared_error = 0.0
    expert_counts = np.zeros(3, dtype=np.int64)
    num_images = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "gt_count", "pred_count", "abs_err"])
        for image_path in tqdm(image_paths, desc="Native 批量推理", leave=False):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logging.warning("无法读取 %s，跳过", image_path)
                continue
            height, width = image_bgr.shape[:2]
            padded, scale, pad_x, pad_y = letterbox_image(image_bgr, imgsz)
            image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                predictions = model(tensor)
            scores = predictions["logits"].sigmoid()[0]
            crop_points = predictions["points"][0]
            sources = predictions["expert_indices"][0]
            keep = scores > args.conf
            if args.count_mode == "soft":
                pred_count = float(scores.sum().item())
            else:
                pred_count = int(keep.sum().item())

            if args.heatmap:
                heat = native_heatmap(scores, model, imgsz)
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                heat = heat[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
                heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
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
                cv2.imwrite(os.path.join(heat_dir, base_name + "_heat.jpg"), overlay)

            crop_points = crop_points[keep].cpu().numpy()
            sources = sources[keep].cpu().numpy()
            pred_points = (
                crop_points - np.array([pad_x, pad_y], dtype=np.float32)
            ) / scale
            pred_points[:, 0] = np.clip(pred_points[:, 0], 0, width - 1)
            pred_points[:, 1] = np.clip(pred_points[:, 1], 0, height - 1)
            gt_points = load_gt_points(
                os.path.join(points_dir, base_name + ".txt"),
                width,
                height,
            )
            gt_count = gt_points.shape[0]
            abs_error = abs(pred_count - gt_count)
            total_abs_error += abs_error
            total_squared_error += abs_error * abs_error
            num_images += 1
            for expert_index in range(3):
                expert_counts[expert_index] += int(
                    (sources == expert_index).sum()
                )
            cv2.imwrite(
                os.path.join(image_out_dir, base_name + "_pred.jpg"),
                draw_result(image_bgr, pred_points, sources, gt_points),
            )
            writer.writerow([base_name, gt_count, pred_count, abs_error])

    if num_images == 0:
        raise RuntimeError("没有成功处理的图像")
    mae = total_abs_error / num_images
    rmse = np.sqrt(total_squared_error / num_images)
    summary = {
        "architecture": "native_multiscale",
        "num_images": num_images,
        "mae": float(mae),
        "rmse": float(rmse),
        "conf": args.conf,
        "count_mode": args.count_mode,
        "heatmap": args.heatmap,
        "imgsz": imgsz,
        "checkpoint": args.checkpoint,
        "expert_usage": {
            f"expert{i}": int(expert_counts[i]) for i in range(3)
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    logging.info("共处理 %d 张验证图像", num_images)
    logging.info("MAE=%.3f RMSE=%.3f", mae, rmse)
    logging.info("专家使用: %s", ", ".join(f"E{i}={int(expert_counts[i])}" for i in range(3)))
    logging.info("汇总: %s", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="native_multiscale 验证集批量推理"
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default="yolo11m.pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--count-mode", choices=("soft", "thresh"), default="soft")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--heat-alpha", type=float, default=0.45)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
