"""Per-image cross-run comparison panels: E0-only / E1-only / E2-only / joint.

对每张 val 图生成一张 2x2 对比图，供人工横向检查“大目标”检出差异：
- 左上: E0-only (P3)      - 右上: E1-only (P4)
- 左下: E2-only (P5)      - 右下: 联合 native (三专家)
每个面板: 原图 + GT(空心圆) + 该 run 的预测点(实心点，按置信度阈值过滤)。

推理协议与正式评估完全一致：tiled 滑窗(imgsz crop) + 余弦窗融合 +
conf 过滤 + NMS，GT 与预测点都在原图像素坐标系，避免整图直接前向
在高分辨率数据集（QNRF/JHU）上与正式 MAE/Recall 口径不一致。
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.data.point_dataset import load_points
from scripts.inference.tiling import run_tiled_inference
from test_each_dataset import load_checkpoint_model

EXPERT_COLORS = {
    0: "#E63232",  # E0/P3 红
    1: "#28C83C",  # E1/P4 绿
    2: "#3278F0",  # E2/P5 蓝
}


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


def load_models(
    checkpoint_paths: list[str],
    weights: str,
    device: str,
) -> list[tuple[str, object, str, int | None]]:
    entries = []
    for path in checkpoint_paths:
        model, metadata = load_checkpoint_model(path, weights, device)
        expert_index = metadata.get("expert_index")
        if expert_index is not None:
            expert_index = int(expert_index)
        routing = "expert_only" if expert_index is not None else "native"
        tag = (
            f"E{expert_index}-only"
            if expert_index is not None
            else "joint"
        )
        model.eval()
        entries.append((tag, model, routing, expert_index))
    return entries


def render_panel(
    ax,
    image: np.ndarray,
    gt: np.ndarray,
    pred_points: np.ndarray,
    pred_scores: np.ndarray,
    pred_sources: np.ndarray,
    routing: str,
    conf_threshold: float,
) -> None:
    ax.imshow(image)
    ax.axis("off")
    if gt.shape[0]:
        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            facecolors="none",
            edgecolors="#FFD700",
            linewidths=0.8,
            s=26,
            marker="o",
        )
    keep = pred_scores > conf_threshold
    if keep.any():
        points = pred_points[keep]
        sources = pred_sources[keep]
        if routing == "expert_only":
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=4,
                c="#FFFFFF",
                edgecolors="none",
            )
        else:
            for expert in range(3):
                mask = sources == expert
                if mask.any():
                    ax.scatter(
                        points[mask, 0],
                        points[mask, 1],
                        s=4,
                        c=EXPERT_COLORS[expert],
                        edgecolors="none",
                    )


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "compare.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到任何 jpg 图片")
    selected = (
        image_paths
        if args.max_images < 0
        else image_paths[: args.max_images]
    )

    entries = load_models(args.checkpoint, args.weights, device)
    logging.info(
        "面板顺序: %s",
        " | ".join(tag for tag, _, _, _ in entries),
    )

    for image_path in tqdm(selected, desc="生成对比图"):
        filename = os.path.basename(image_path)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logging.warning("无法读取 %s，跳过", image_path)
            continue
        height, width = image_bgr.shape[:2]
        gt = load_points(
            os.path.join(
                points_dir,
                os.path.splitext(filename)[0] + ".txt",
            ),
            width,
            height,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        cols = min(len(entries), 2)
        rows = (len(entries) + 1) // 2
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(9 * cols, 9 * rows),
            dpi=150,
        )
        axes = np.asarray(axes).reshape(-1)

        with torch.inference_mode():
            for panel_index, (tag, model, routing, expert_index) in enumerate(
                entries
            ):
                result = run_tiled_inference(
                    model,
                    image_bgr,
                    device,
                    args.imgsz,
                    overlap=args.overlap,
                    tile_batch_size=args.tile_batch_size,
                    conf_threshold=args.conf_threshold,
                    nms_radius=args.nms_radius,
                    routing_mode=routing,
                    expert_index=expert_index,
                )
                ax = axes[panel_index]
                render_panel(
                    ax,
                    image_rgb,
                    gt,
                    result.points,
                    result.scores,
                    result.sources,
                    routing,
                    args.conf_threshold,
                )
                ax.set_title(
                    f"{tag} | GT={gt.shape[0]} Pred={result.count:.1f}",
                    fontsize=13,
                    fontweight="bold",
                    pad=6,
                )

        for extra in range(len(entries), axes.size):
            axes[extra].axis("off")

        fig.suptitle(
            f"{filename} ({width}x{height}, tiled crop={args.imgsz}, "
            f"conf>{args.conf_threshold})",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = os.path.join(
            args.out_dir, f"{os.path.splitext(filename)[0]}.jpg"
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    logging.info("对比图输出目录: %s", args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同一张 val 图的 E0/E1/E2/联合 横向对比面板（tiled 推理协议）"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="依次作为面板 (建议 E0-only/E1-only/E2-only/joint)",
    )
    parser.add_argument("--data-root", type=str, default="datasets/shanghaitech_AB")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="tiled crop size（与正式评估一致）",
    )
    parser.add_argument("--out-dir", type=str, default="runs/ablation_eval/compare")
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=4,
        help="tiled 推理 NMS 去重半径；默认 4px 只去重 overlap 的重复候选",
    )
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="生成前 N 张；-1 生成全部",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    main(parse_args())
