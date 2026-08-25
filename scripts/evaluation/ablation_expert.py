"""Single-expert ablation evaluation: per-GT nearest distance + Recall@r + count MAE.

支持两类 checkpoint:
- 联合训练 (native): 输出全部三专家候选，按 expert_indices 分专家统计。
- 单专家消融 (expert_only, 由 checkpoint config["expert_index"] 标记):
  只输出被保留专家的候选。

推理协议与正式验证完全一致（tiled 滑窗 + 余弦窗融合，原图坐标）：

    原始图像 → tiled inference → 原图坐标 predictions → 原图坐标 GT
        → 逐 GT 最近距离 d(g) = min_j |p_j - g|

统计：Recall@8/16/32、mean/median 最近距离（仅有限距离）、各专家计数 MAE
（专家计数 = 该专家层 fused soft count）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
from dataclasses import dataclass, field

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import cv2
import numpy as np
import torch
from tqdm import tqdm

from scripts.data.point_dataset import load_points
from scripts.inference.tiling import run_tiled_inference
from scripts.visualization.plot_utils import (
    create_moe_comparison_figure,
    save_figure,
)
from test_each_dataset import load_checkpoint_model

RADII = (8, 16, 32)


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


@dataclass
class ExpertStats:
    """按专家累积的 GT 级定位/计数统计。

    nearest_dists 是逐 GT 数组（无候选时对应 inf），recall 分母为全部 GT。
    """

    images: int = 0
    gt_total: int = 0
    dist_sum: float = 0.0
    dist_list: list[float] = field(default_factory=list)
    recall_counts: dict[float, int] = field(
        default_factory=lambda: {radius: 0 for radius in RADII}
    )
    count_abs_error: float = 0.0
    count_squared_error: float = 0.0
    count_bias: float = 0.0

    def update(
        self,
        nearest_dists: np.ndarray,
        pred_count: float,
    ) -> None:
        """nearest_dists: [N_gt] 每个 GT 到该专家最近预测点的距离（可含 inf）。"""
        self.images += 1
        if nearest_dists.size > 0:
            self.gt_total += int(nearest_dists.size)
            finite = nearest_dists[np.isfinite(nearest_dists)]
            if finite.size > 0:
                self.dist_sum += float(finite.sum())
                self.dist_list.extend(float(value) for value in finite)
            for radius in RADII:
                self.recall_counts[radius] += int(
                    (nearest_dists <= radius).sum()
                )
        error = pred_count - int(nearest_dists.size)
        self.count_abs_error += abs(error)
        self.count_squared_error += error * error
        self.count_bias += error

    def summary(self) -> dict[str, float | int | None]:
        if self.images == 0:
            return {"images": 0, "mae": None, "rmse": None, "bias": None}
        result: dict[str, float | int | None] = {
            "images": self.images,
            "mae": self.count_abs_error / self.images,
            "rmse": (self.count_squared_error / self.images) ** 0.5,
            "bias": self.count_bias / self.images,
        }
        if self.gt_total > 0:
            distances = np.asarray(self.dist_list)
            result["gt_total"] = self.gt_total
            result["mean_dist_px"] = (
                float(distances.mean()) if distances.size > 0 else None
            )
            result["median_dist_px"] = (
                float(np.median(distances)) if distances.size > 0 else None
            )
            for radius in RADII:
                result[f"recall@{radius}px"] = (
                    self.recall_counts[radius] / self.gt_total
                )
        else:
            result["gt_total"] = 0
            result["mean_dist_px"] = None
            result["median_dist_px"] = None
            for radius in RADII:
                result[f"recall@{radius}px"] = None
        return result


def evaluate_checkpoint(
    args,
    checkpoint_path: str,
    device: str,
) -> dict[str, object]:
    model, metadata = load_checkpoint_model(
        checkpoint_path,
        args.weights,
        device,
    )
    expert_index = metadata.get("expert_index")
    if expert_index is not None:
        expert_index = int(expert_index)
    routing_mode = "expert_only" if expert_index is not None else "native"
    crop_size = args.imgsz or int(metadata["crop_size"])
    output_strides = [
        int(value) for value in model.point_head.output_strides
    ]

    image_dir = os.path.join(args.data_root, "images", args.split)
    points_dir = os.path.join(args.data_root, "points", args.split)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到任何 jpg 图片")

    stats = [ExpertStats() for _ in range(3)]
    run_tag = os.path.basename(os.path.dirname(checkpoint_path))
    per_image_rows: list[dict[str, object]] = []
    vis_dir = os.path.join(args.out_dir, "vis", run_tag)
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        for image_path in tqdm(
            image_paths,
            desc=f"{run_tag} 评估中",
            leave=False,
        ):
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logging.warning("无法读取 %s，跳过", image_path)
                continue
            height, width = image_bgr.shape[:2]
            gt = load_points(
                os.path.join(
                    points_dir,
                    os.path.splitext(os.path.basename(image_path))[0]
                    + ".txt",
                ),
                width,
                height,
            )
            gt_count = gt.shape[0]
            result = run_tiled_inference(
                model,
                image_bgr,
                device,
                crop_size,
                overlap=args.overlap,
                tile_batch_size=args.batch_size,
                conf_threshold=args.conf,
                nms_radius=args.nms_radius,
                routing_mode=routing_mode,
                expert_index=expert_index,
            )
            pred_points = result.points
            pred_sources = result.sources
            pred_scores = result.scores
            pred_count = float(result.count)

            row: dict[str, object] = {
                "image": len(per_image_rows),
                "filename": os.path.basename(image_path),
                "gt_count": gt_count,
            }
            expert_counts = [0.0, 0.0, 0.0]
            expert_dists: list[np.ndarray | None] = [
                None,
                None,
                None,
            ]
            for expert in range(3):
                expert_counts[expert] = float(
                    result.level_counts.get(
                        output_strides[expert], 0.0
                    )
                )
                mask = pred_sources == expert
                expert_points = pred_points[mask]
                if gt_count > 0:
                    if expert_points.shape[0] > 0:
                        pair = torch.cdist(
                            torch.from_numpy(expert_points).float(),
                            torch.from_numpy(gt).float(),
                            p=2,
                        )
                        expert_dists[expert] = (
                            pair.min(dim=0).values.numpy()
                        )
                    else:
                        expert_dists[expert] = np.full(
                            gt_count,
                            np.inf,
                            dtype=np.float32,
                        )
                if (
                    routing_mode == "expert_only"
                    and expert != expert_index
                ):
                    # 未参与专家：无候选，不累积统计
                    continue
                stats[expert].update(
                    expert_dists[expert]
                    if expert_dists[expert] is not None
                    else np.empty(0, dtype=np.float32),
                    expert_counts[expert],
                )

            for expert in range(3):
                row[f"count_E{expert}"] = expert_counts[expert]
                if expert_dists[expert] is None:
                    for radius in RADII:
                        row[f"recall_E{expert}@{radius}px"] = None
                else:
                    finite = expert_dists[expert][
                        np.isfinite(expert_dists[expert])
                    ]
                    for radius in RADII:
                        row[f"recall_E{expert}@{radius}px"] = float(
                            (expert_dists[expert] <= radius).mean()
                        )
                    row[f"meanD_E{expert}"] = (
                        float(finite.mean()) if finite.size > 0 else None
                    )
                    row[f"medianD_E{expert}"] = (
                        float(np.median(finite))
                        if finite.size > 0
                        else None
                    )
            per_image_rows.append(row)

            if len(per_image_rows) <= args.vis_images:
                image_rgb = cv2.cvtColor(
                    image_bgr, cv2.COLOR_BGR2RGB
                )
                figure = create_moe_comparison_figure(
                    image=image_rgb,
                    gt_points=gt,
                    pred_points=pred_points,
                    pred_routes=pred_sources,
                    pred_scores=pred_scores,
                    gt_count=float(gt_count),
                    pred_count=pred_count,
                    title=f"{run_tag} | {row['filename']} | "
                    f"GT={gt_count} E0={expert_counts[0]:.1f} "
                    f"E1={expert_counts[1]:.1f} E2={expert_counts[2]:.1f}",
                    conf_threshold=args.vis_conf,
                )
                save_figure(
                    figure,
                    os.path.join(
                        vis_dir,
                        f"{len(per_image_rows) - 1:03d}_"
                        f"{os.path.splitext(str(row['filename']))[0]}.jpg",
                    ),
                )

    return {
        "checkpoint": checkpoint_path,
        "run": run_tag,
        "routing_mode": routing_mode,
        "expert_index": expert_index,
        "epoch": metadata.get("epoch"),
        "per_expert": {
            f"E{expert}": stats[expert].summary()
            for expert in range(3)
        },
        "per_image": per_image_rows,
    }


def format_recall_line(
    result: dict[str, object],
    expert: int,
) -> str:
    summary = result["per_expert"][f"E{expert}"]
    if summary["images"] == 0:
        return f"E{expert}: (未参与，无候选)"
    line = (
        f"E{expert}: MAE={summary['mae']:.1f} "
        f"RMSE={summary['rmse']:.1f} bias={summary['bias']:+.1f}"
    )
    if summary.get("gt_total"):
        line += (
            f" meanD={summary['mean_dist_px']:.1f}px "
            f"medD={summary['median_dist_px']:.1f}px"
        )
        for radius in RADII:
            recall = summary[f"recall@{radius}px"]
            line += (
                f" R@{radius}={recall * 100:.1f}%"
                if recall is not None
                else f" R@{radius}=N/A"
            )
    return line


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "ablation.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    all_results = []
    for checkpoint_path in args.checkpoint:
        result = evaluate_checkpoint(args, checkpoint_path, device)
        all_results.append(result)
        logging.info("== %s (routing=%s) ==", result["run"], result["routing_mode"])
        for expert in range(3):
            logging.info("  %s", format_recall_line(result, expert))

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "radii_px": list(RADII),
                "count_metric": "level_fused_soft_count (tiled)",
                "inference": "tiled_cosine",
                "conf_threshold": args.conf,
                "nms_radius": args.nms_radius,
                "results": [
                    {
                        key: value
                        for key, value in result.items()
                        if key != "per_image"
                    }
                    for result in all_results
                ],
            },
            summary_file,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = os.path.join(args.out_dir, "per_image.csv")
    headers = (
        ["run", "filename", "gt_count"]
        + [f"count_E{expert}" for expert in range(3)]
        + [
            f"recall_E{expert}@{radius}px"
            for expert in range(3)
            for radius in RADII
        ]
        + [
            f"meanD_E{expert}"
            for expert in range(3)
        ]
        + [
            f"medianD_E{expert}"
            for expert in range(3)
        ]
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        for result in all_results:
            for row in result["per_image"]:
                writer.writerow(
                    [
                        result["run"],
                        row["filename"],
                        row["gt_count"],
                    ]
                    + [row[f"count_E{expert}"] for expert in range(3)]
                    + [
                        row[f"recall_E{expert}@{radius}px"]
                        for expert in range(3)
                        for radius in RADII
                    ]
                    + [row[f"meanD_E{expert}"] for expert in range(3)]
                    + [row[f"medianD_E{expert}"] for expert in range(3)]
                )
    logging.info("汇总: %s", summary_path)
    logging.info("逐图明细: %s", csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单专家消融评估：tiled 推理 + 逐 GT 最近距离、Recall@r、计数 MAE"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="checkpoint 路径，可重复指定；expert_only 由 config 自动识别",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/shanghaitech_AB",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="runs/ablation_eval")
    parser.add_argument("--vis-images", type=int, default=0)
    parser.add_argument("--vis-conf", type=float, default=0.3)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="tiled 推理的候选置信度阈值（用于定位/Recall；计数始终用软计数）",
    )
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=4,
        help="tiled 推理 NMS 去重半径；默认 4px 只去重重叠 tile 的重复候选",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    main(parse_args())
