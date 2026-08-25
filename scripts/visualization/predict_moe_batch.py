"""Batch inference and visualization for the native_multiscale head."""

from __future__ import annotations

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

from scripts.data.point_dataset import load_points
from scripts.inference.tiling import run_tiled_inference
from scripts.visualization.predict_moe import (
    EXPERT_COLORS,
    load_model,
    overlay_probability,
    resolve_inference_settings,
)

GT_COLOR = (0, 255, 255)
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


def _resolve_routing(args, metadata: dict[str, object]) -> tuple[str, int | None]:
    routing_mode = str(metadata["routing_mode"])
    expert_index = metadata.get("expert_index")
    if expert_index is not None:
        expert_index = int(expert_index)
    if args.expert_index is not None:
        expert_index = int(args.expert_index)
        routing_mode = "expert_only"
        logging.info(
            "命令行覆盖推理路由: routing_mode=%s expert_index=%s",
            routing_mode,
            expert_index,
        )
    return routing_mode, expert_index


def _write_heatmaps(
    heat_dir: str,
    base_name: str,
    image_bgr: np.ndarray,
    prob_map: np.ndarray,
    heat_alpha: float,
) -> None:
    os.makedirs(heat_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(heat_dir, base_name + "_prob.jpg"),
        overlay_probability(image_bgr, prob_map, heat_alpha),
    )
    normalized = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(
        os.path.join(heat_dir, base_name + "_prob_contrast.png"),
        cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET),
    )
    fixed = np.clip(prob_map, 0.0, 1.0)
    cv2.imwrite(
        os.path.join(heat_dir, base_name + "_prob_fixed.png"),
        (fixed * 255).astype(np.uint8),
    )


def predict_batch(args: argparse.Namespace) -> dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "predict.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_model(args.weights, args.checkpoint, device)
    routing_mode, expert_index = _resolve_routing(args, metadata)
    imgsz = resolve_inference_settings(args.checkpoint, imgsz=args.imgsz)
    logging.info(
        "推理设置: imgsz=%d routing_mode=%s expert_index=%s device=%s",
        imgsz,
        routing_mode,
        expert_index,
        device,
    )

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到任何 jpg 图片")

    image_out_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_out_dir, exist_ok=True)
    heat_dir = os.path.join(args.out_dir, "heatmaps")
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_gt = 0
    total_pred = 0.0
    expert_counts = np.zeros(3, dtype=np.int64)
    num_images = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "gt_count", "pred_count", "abs_err"])
        with torch.inference_mode():
            for image_path in tqdm(
                image_paths,
                desc="Native 批量推理",
                leave=False,
            ):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    logging.warning("无法读取 %s，跳过", image_path)
                    continue
                height, width = image_bgr.shape[:2]
                result = run_tiled_inference(
                    model,
                    image_bgr,
                    device,
                    imgsz,
                    overlap=args.overlap,
                    tile_batch_size=args.tile_batch_size,
                    conf_threshold=args.conf,
                    routing_mode=routing_mode,
                    expert_index=expert_index,
                )
                pred_count = (
                    float(result.count)
                    if args.count_mode == "soft"
                    else int(len(result.points))
                )
                gt_points = load_points(
                    os.path.join(points_dir, base_name + ".txt"),
                    width,
                    height,
                )
                gt_count = int(gt_points.shape[0])
                error = pred_count - gt_count
                abs_error = abs(error)
                total_abs_error += abs_error
                total_squared_error += error * error
                total_gt += gt_count
                total_pred += pred_count
                num_images += 1
                for expert in range(3):
                    expert_counts[expert] += int(
                        (result.sources == expert).sum()
                    )

                if args.heatmap:
                    _write_heatmaps(
                        heat_dir,
                        base_name,
                        image_bgr,
                        result.prob_map,
                        args.heat_alpha,
                    )
                cv2.imwrite(
                    os.path.join(image_out_dir, base_name + "_pred.jpg"),
                    draw_result(
                        image_bgr,
                        result.points,
                        result.sources,
                        gt_points,
                    ),
                )
                writer.writerow(
                    [
                        base_name,
                        gt_count,
                        f"{pred_count:.3f}",
                        f"{abs_error:.3f}",
                    ]
                )

    if num_images == 0:
        raise RuntimeError("没有成功处理的图像")
    mae = total_abs_error / num_images
    rmse = float(np.sqrt(total_squared_error / num_images))
    summary: dict[str, object] = {
        "architecture": NATIVE_ARCHITECTURE,
        "num_images": num_images,
        "gt_total": total_gt,
        "pred_total": total_pred,
        "mae": float(mae),
        "rmse": rmse,
        "conf": args.conf,
        "count_mode": args.count_mode,
        "heatmap": args.heatmap,
        "imgsz": imgsz,
        "inference": "tiled_cosine",
        "count_metric": (
            "native_sum_sigmoid_tiled_padding_safe"
            if args.count_mode == "soft"
            else "thresholded_owned_points"
        ),
        "overlap": args.overlap,
        "tile_batch_size": args.tile_batch_size,
        "routing_mode": routing_mode,
        "expert_index": expert_index,
        "expert_usage": {
            f"expert{i}": int(expert_counts[i]) for i in range(3)
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    logging.info("共处理 %d 张验证图像", num_images)
    logging.info("MAE=%.3f RMSE=%.3f", mae, rmse)
    logging.info(
        "专家使用: %s",
        ", ".join(
            f"E{i}={int(expert_counts[i])}" for i in range(3)
        ),
    )
    logging.info("汇总: %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="native_multiscale 验证集批量推理"
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument(
        "--count-mode",
        choices=("soft", "thresh"),
        default="soft",
    )
    parser.add_argument("--heat-alpha", type=float, default=0.45)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/native_multiscale/predict_batch",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=True,
        help="保存三种概率热力图（默认开启）",
    )
    parser.add_argument(
        "--no-heatmap",
        dest="heatmap",
        action="store_false",
        help="不保存概率热力图",
    )
    parser.add_argument(
        "--expert-index",
        type=int,
        default=None,
        help="覆盖 checkpoint 记录的 expert_index（0/1/2）；缺省自动恢复",
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
