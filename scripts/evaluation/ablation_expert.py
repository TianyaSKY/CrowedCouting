"""Single-expert ablation evaluation: per-GT nearest distance + Recall@r + count MAE.

支持两类 checkpoint:
- 联合训练 (native): 输出全部三专家候选，按 expert_indices 分专家统计。
- 单专家消融 (expert_only, 由 checkpoint config["expert_index"] 标记):
  只输出被保留专家的候选。

推理协议与正式验证完全一致（tiled 滑窗 + 余弦窗融合 + tile ownership，原图坐标）：

    原始图像 → tiled inference → 原图坐标 predictions → 原图坐标 GT
        → 逐 GT 最近距离 d(g) = min_j ||p_j - g||₂

统计：Recall@8/16/32、mean/median 最近距离（Euclidean L2，仅有限距离）、
各专家计数 MAE（专家计数 = 该专家层 fused soft count）。

另按 GT 邻域间距 heuristic（与最近 3 个 GT 距离的 median）切成
dense_small_proxy / medium_proxy / sparse_large_proxy 三桶，分别统计
Recall@8/16/32、最近距离 mean/median 与 GT 邻域最大置信度。
该 proxy 反映局部标注密度，不是 ground-truth object size，不能单独
支持“真正的大目标 Recall 提升”的结论。
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

# GT 邻域原始置信度统计半径（不受阈值/NMS 影响）。
RAW_CONF_RADII = (16, 32)
RAW_CONF_CANDIDATE_CHUNK_SIZE = 4096
RAW_CONF_GT_CHUNK_SIZE = 4096
# scale proxy 的最近邻数量；它是局部密度 heuristic，不是目标尺寸。
SCALE_K = 3
SCALE_BIN_NAMES = (
    "dense_small_proxy",
    "medium_proxy",
    "sparse_large_proxy",
)


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


def scale_proxies(gt: np.ndarray) -> np.ndarray:
    """每个 GT 的局部间距 proxy：最近 SCALE_K 个 GT 距离的 median。

    dense_small_proxy 通常表示更密的远景/小头布局，sparse_large_proxy
    通常表示更稀的近景/大头布局；但透视、构图和标注密度都会影响它，
    因此它不是 ground-truth object size。
    """
    count = gt.shape[0]
    proxies = np.full(count, np.inf, dtype=np.float64)
    if count < 2:
        return proxies
    distances = np.linalg.norm(
        gt[:, None, :] - gt[None, :, :], axis=-1
    )
    for index in range(count):
        neighbors = np.sort(distances[index])
        neighbors = neighbors[neighbors > 1e-6]
        if neighbors.size == 0:
            continue
        proxies[index] = float(
            np.median(neighbors[:SCALE_K])
        )
    return proxies


def scale_bin_thresholds(
    proxies: np.ndarray,
) -> tuple[float, float]:
    """按 33.3%/66.7% 分位切分 proxy；inf 归入 sparse_large_proxy。"""
    finite = proxies[np.isfinite(proxies)]
    if finite.size < 2:
        return 0.0, 0.0
    return (
        float(np.quantile(finite, 1.0 / 3.0)),
        float(np.quantile(finite, 2.0 / 3.0)),
    )


def assign_scale_bin(
    proxy: float,
    thresholds: tuple[float, float],
) -> str:
    low, high = thresholds
    if proxy <= low:
        return "dense_small_proxy"
    if proxy <= high:
        return "medium_proxy"
    return "sparse_large_proxy"


def _raw_max_confidences(
    raw_points: np.ndarray,
    raw_scores: np.ndarray,
    gt: np.ndarray,
    candidate_chunk_size: int = RAW_CONF_CANDIDATE_CHUNK_SIZE,
    gt_chunk_size: int = RAW_CONF_GT_CHUNK_SIZE,
) -> np.ndarray:
    """返回每个 GT 在各诊断半径内的最大原始置信度。

    候选和 GT 均分块，避免为一张大图构造
    ``[all_raw_candidates, all_gt]`` 距离矩阵。
    """
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if gt_chunk_size <= 0:
        raise ValueError("gt_chunk_size must be positive")

    max_confs = np.zeros(
        (gt.shape[0], len(RAW_CONF_RADII)),
        dtype=np.float32,
    )
    if raw_points.shape[0] == 0 or gt.shape[0] == 0:
        return max_confs
    if raw_points.shape[0] != raw_scores.shape[0]:
        raise ValueError("raw_points and raw_scores must have equal length")

    points_tensor = torch.from_numpy(
        np.ascontiguousarray(raw_points, dtype=np.float32)
    )
    scores_tensor = torch.from_numpy(
        np.ascontiguousarray(raw_scores, dtype=np.float32)
    )
    gt_tensor = torch.from_numpy(
        np.ascontiguousarray(gt, dtype=np.float32)
    )
    max_conf_tensor = torch.zeros(
        (gt.shape[0], len(RAW_CONF_RADII)),
        dtype=torch.float32,
    )
    for candidate_start in range(
        0,
        points_tensor.shape[0],
        candidate_chunk_size,
    ):
        candidate_end = min(
            candidate_start + candidate_chunk_size,
            points_tensor.shape[0],
        )
        candidate_points = points_tensor[
            candidate_start:candidate_end
        ]
        candidate_scores = scores_tensor[
            candidate_start:candidate_end
        ]
        for gt_start in range(0, gt_tensor.shape[0], gt_chunk_size):
            gt_end = min(gt_start + gt_chunk_size, gt_tensor.shape[0])
            distances = torch.cdist(
                candidate_points,
                gt_tensor[gt_start:gt_end],
                p=2,
            )
            for radius_index, radius in enumerate(RAW_CONF_RADII):
                nearby = distances <= radius
                if not bool(nearby.any()):
                    continue
                chunk_max = torch.where(
                    nearby,
                    candidate_scores[:, None],
                    0.0,
                ).amax(dim=0)
                max_conf_tensor[gt_start:gt_end, radius_index] = (
                    torch.maximum(
                        max_conf_tensor[gt_start:gt_end, radius_index],
                        chunk_max,
                    )
                )
    return max_conf_tensor.numpy()


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


@dataclass
class ScaleBinStats:
    """按 GT scale proxy 分桶的定位/置信度统计（run 级候选）。

    recall 来自 conf>=args.conf 的候选（与 Recall@r 口径一致）；
    RawMaxConf 来自阈值/NMS 之前的全部原始候选。
    """

    gt_total: int = 0
    dist_list: list[float] = field(default_factory=list)
    recall_counts: dict[float, int] = field(
        default_factory=lambda: {radius: 0 for radius in RADII}
    )
    raw_conf_sums: dict[float, float] = field(
        default_factory=lambda: {
            radius: 0.0 for radius in RAW_CONF_RADII
        }
    )

    def update(
        self,
        nearest_dist: float,
        raw_max_conf_16: float,
        raw_max_conf_32: float,
    ) -> None:
        self.gt_total += 1
        if np.isfinite(nearest_dist):
            self.dist_list.append(float(nearest_dist))
            for radius in RADII:
                if nearest_dist <= radius:
                    self.recall_counts[radius] += 1
        self.raw_conf_sums[16] += float(raw_max_conf_16)
        self.raw_conf_sums[32] += float(raw_max_conf_32)

    def summary(self) -> dict[str, float | int | None]:
        if self.gt_total == 0:
            return {
                "gt_total": 0,
                "recall@8px": None,
                "recall@16px": None,
                "recall@32px": None,
                "mean_dist_px": None,
                "median_dist_px": None,
                "raw_max_conf@16px": None,
                "raw_max_conf@32px": None,
            }
        distances = np.asarray(self.dist_list)
        return {
            "gt_total": self.gt_total,
            "recall@8px": self.recall_counts[8] / self.gt_total,
            "recall@16px": self.recall_counts[16] / self.gt_total,
            "recall@32px": self.recall_counts[32] / self.gt_total,
            "mean_dist_px": (
                float(distances.mean()) if distances.size > 0 else None
            ),
            "median_dist_px": (
                float(np.median(distances)) if distances.size > 0 else None
            ),
            "raw_max_conf@16px": (
                self.raw_conf_sums[16] / self.gt_total
            ),
            "raw_max_conf@32px": (
                self.raw_conf_sums[32] / self.gt_total
            ),
        }


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

    # 全 run 的逐 GT 记录: (scale_proxy, 最近距离, RawMaxConf@16, RawMaxConf@32)
    gt_scale_records: list[tuple[float, float, float, float]] = []
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
                return_raw_candidates=True,
            )
            pred_points = result.points
            pred_sources = result.sources
            pred_scores = result.scores
            raw_points = result.raw_points
            raw_scores = result.raw_scores
            pred_count = float(result.count)

            row: dict[str, object] = {
                "image": len(per_image_rows),
                "filename": os.path.basename(image_path),
                "gt_count": gt_count,
            }
            if gt_count > 0:
                proxies = scale_proxies(gt)
                row["gt_proxy_median"] = (
                    float(np.median(proxies[np.isfinite(proxies)]))
                    if np.isfinite(proxies).any()
                    else None
                )
                if pred_points.shape[0] > 0:
                    pair = torch.cdist(
                        torch.from_numpy(pred_points).float(),
                        torch.from_numpy(gt).float(),
                        p=2,
                    )
                    nearest_dists = pair.min(dim=0).values.numpy()
                else:
                    nearest_dists = np.full(
                        gt_count, np.inf, dtype=np.float32
                    )
                # 原始候选（阈值/NMS 之前）的 GT 邻域最大置信度：
                # 不受 conf 过滤影响，能区分 0.02 / 0.08 / 0.30 的差异。
                raw_max_confs = _raw_max_confidences(
                    raw_points,
                    raw_scores,
                    gt,
                )
                for gt_index in range(gt_count):
                    gt_scale_records.append(
                        (
                            float(proxies[gt_index]),
                            float(nearest_dists[gt_index]),
                            float(raw_max_confs[gt_index, 0]),
                            float(raw_max_confs[gt_index, 1]),
                        )
                    )
            else:
                row["gt_proxy_median"] = None
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
                    row[f"meanD_L2_E{expert}"] = None
                    row[f"medianD_L2_E{expert}"] = None
                else:
                    finite = expert_dists[expert][
                        np.isfinite(expert_dists[expert])
                    ]
                    for radius in RADII:
                        row[f"recall_E{expert}@{radius}px"] = float(
                            (expert_dists[expert] <= radius).mean()
                        )
                    row[f"meanD_L2_E{expert}"] = (
                        float(finite.mean()) if finite.size > 0 else None
                    )
                    row[f"medianD_L2_E{expert}"] = (
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

    # 按局部 GT 间距 heuristic 分桶；不把 proxy 当作真实目标尺寸。
    proxies = np.asarray(
        [record[0] for record in gt_scale_records],
        dtype=np.float64,
    )
    thresholds = scale_bin_thresholds(proxies)
    bin_stats = {
        name: ScaleBinStats() for name in SCALE_BIN_NAMES
    }
    for proxy, nearest_dist, raw_conf_16, raw_conf_32 in (
        gt_scale_records
    ):
        bin_stats[
            assign_scale_bin(proxy, thresholds)
        ].update(nearest_dist, raw_conf_16, raw_conf_32)
    scale_bins = {
        name: bin_stats[name].summary() for name in SCALE_BIN_NAMES
    }
    scale_bins["thresholds"] = {
        "p33": thresholds[0],
        "p66": thresholds[1],
        "scale_proxy": "median distance to 3 nearest GT",
        "scale_proxy_semantics": (
            "local GT spacing heuristic; not ground-truth object size"
        ),
    }

    return {
        "checkpoint": checkpoint_path,
        "run": run_tag,
        "routing_mode": routing_mode,
        "expert_index": expert_index,
        "epoch": metadata.get("epoch"),
        "nearest_distance_metric": "euclidean_l2_px",
        "scale_proxy": "median distance to 3 nearest GT",
        "scale_proxy_is_ground_truth_size": False,
        "per_expert": {
            f"E{expert}": stats[expert].summary()
            for expert in range(3)
        },
        "scale_bins": scale_bins,
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
            f" meanD_L2={summary['mean_dist_px']:.1f}px "
            f"medD_L2={summary['median_dist_px']:.1f}px"
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
        scale_bins = result.get("scale_bins", {})
        if scale_bins:
            thresholds = scale_bins.get("thresholds", {})
            logging.info(
                "  scale bins (proxy=median dist to 3 nearest GT; "
                "density heuristic, not object size; p33=%.1fpx p66=%.1fpx):",
                thresholds.get("p33", 0.0),
                thresholds.get("p66", 0.0),
            )
            for name in SCALE_BIN_NAMES:
                summary = scale_bins.get(name, {})
                line = (
                    f"    {name:<12} n={summary.get('gt_total', 0):>5}"
                )
                for radius in RADII:
                    recall = summary.get(f"recall@{radius}px")
                    line += (
                        f" R@{radius}={recall * 100:.1f}%"
                        if recall is not None
                        else f" R@{radius}=N/A"
                    )
                mean_dist = summary.get("mean_dist_px")
                median_dist = summary.get("median_dist_px")
                line += (
                    f" meanD_L2={mean_dist:.1f}px"
                    if mean_dist is not None
                    else " meanD_L2=N/A"
                )
                line += (
                    f" medD_L2={median_dist:.1f}px"
                    if median_dist is not None
                    else " medD_L2=N/A"
                )
                for radius in RAW_CONF_RADII:
                    raw_conf = summary.get(f"raw_max_conf@{radius}px")
                    line += (
                        f" RawMaxConf@{radius}={raw_conf:.3f}"
                        if raw_conf is not None
                        else f" RawMaxConf@{radius}=N/A"
                    )
                logging.info("%s", line)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "radii_px": list(RADII),
                "count_metric": "native_sum_sigmoid_tiled_padding_safe",
                "inference": "tiled_cosine",
                "nearest_distance_metric": "euclidean_l2_px",
                "scale_proxy_semantics": (
                    "median distance to 3 nearest GT; "
                    "local density heuristic, not object size"
                ),
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
        + [f"meanD_L2_E{expert}" for expert in range(3)]
        + [f"medianD_L2_E{expert}" for expert in range(3)]
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
                    + [
                        row[f"meanD_L2_E{expert}"] for expert in range(3)
                    ]
                    + [
                        row[f"medianD_L2_E{expert}"] for expert in range(3)
                    ]
                    + [row["gt_proxy_median"]]
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
