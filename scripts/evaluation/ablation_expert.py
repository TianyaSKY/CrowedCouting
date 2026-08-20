"""Single-expert ablation evaluation: per-GT nearest distance + Recall@r + count MAE.

支持两类 checkpoint:
- 联合训练 (native): 输出全部三专家候选，按 expert_indices 分专家统计。
- 单专家消融 (expert_only, 由 checkpoint config["expert_index"] 标记):
  只输出被保留专家的候选。

对每个 GT 点计算到各专家最近预测点的距离 d_e(g) = min_j |p_{e,j} - g|，
统计 Recall@8/16/32、mean/median 最近距离，以及各专家的计数 MAE
(单专家计数 = 该专家候选 sum(sigmoid(logits)))。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass, field

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data.point_dataset import PointDataset, point_collate_fn
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
    """按专家累积的 GT 级定位/计数统计。"""

    gt_total: int = 0
    images: int = 0
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
        gt_count: int,
        nearest_dists: np.ndarray,
        pred_count: float,
    ) -> None:
        self.images += 1
        if gt_count > 0 and nearest_dists.size > 0:
            self.gt_total += gt_count
            self.dist_sum += float(nearest_dists.sum())
            self.dist_list.extend(
                float(value) for value in nearest_dists
            )
            for radius in RADII:
                self.recall_counts[radius] += int(
                    (nearest_dists <= radius).sum()
                )
        error = pred_count - gt_count
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
            result["mean_dist_px"] = float(distances.mean())
            result["median_dist_px"] = float(np.median(distances))
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
    config = metadata["config"]
    assert isinstance(config, dict)
    expert_index = config.get("expert_index")
    if expert_index is not None:
        expert_index = int(expert_index)
    routing_mode = "expert_only" if expert_index is not None else "native"

    crop_size = args.imgsz or int(metadata["crop_size"])
    dataset = PointDataset(
        args.data_root,
        split=args.split,
        crop_size=crop_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=point_collate_fn,
    )

    stats = [ExpertStats() for _ in range(3)]
    run_tag = os.path.basename(os.path.dirname(checkpoint_path))
    per_image_rows: list[dict[str, object]] = []
    vis_dir = os.path.join(args.out_dir, "vis", run_tag)
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(loader, desc=f"{run_tag} 评估中", leave=False)
        ):
            images = batch["img"].to(device)
            gt_points = [p.to(device) for p in batch["points"]]
            predictions = model(
                images,
                routing_mode=routing_mode,
                expert_index=expert_index,
            )
            logits = predictions["logits"]
            points = predictions["points"]
            indices = predictions["expert_indices"]
            image_paths = batch.get("image_paths", [])

            for image_index in range(images.shape[0]):
                gt = gt_points[image_index]
                gt_count = gt.shape[0]
                row: dict[str, object] = {
                    "image": batch_index * args.batch_size + image_index,
                    "filename": (
                        os.path.basename(image_paths[image_index])
                        if isinstance(image_paths, list)
                        and image_index < len(image_paths)
                        else f"img_{batch_index}_{image_index}"
                    ),
                    "gt_count": gt_count,
                }
                expert_counts = [0.0, 0.0, 0.0]
                expert_dists: list[np.ndarray | None] = [
                    None,
                    None,
                    None,
                ]
                for expert in range(3):
                    mask = (
                        indices[image_index] == expert
                    )
                    expert_logits = logits[image_index][mask]
                    expert_points = points[image_index][mask]
                    expert_counts[expert] = float(
                        expert_logits.sigmoid().sum()
                    )
                    if gt_count > 0 and expert_points.shape[0] > 0:
                        pair = torch.cdist(
                            expert_points.float(),
                            gt.float(),
                            p=2,
                        )
                        expert_dists[expert] = (
                            pair.min(dim=0).values.cpu().numpy()
                        )
                    if routing_mode == "expert_only" and expert != expert_index:
                        # 未参与专家：无候选，不累积统计
                        continue
                    stats[expert].update(
                        gt_count,
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
                        for radius in RADII:
                            row[f"recall_E{expert}@{radius}px"] = float(
                                (expert_dists[expert] <= radius).mean()
                            )
                per_image_rows.append(row)

                if len(per_image_rows) <= args.vis_images:
                    image = images[image_index]
                    pred_mask = indices[image_index] == (
                        expert_index if expert_index is not None else -1
                    )
                    if expert_index is None:
                        # 联合 checkpoint：叠加图分专家着色，但只画一个整池图即可
                        vis_points = points[image_index]
                        vis_routes = indices[image_index]
                    else:
                        vis_points = points[image_index][pred_mask]
                        vis_routes = indices[image_index][pred_mask]
                    figure = create_moe_comparison_figure(
                        image=image,
                        gt_points=gt,
                        pred_points=vis_points,
                        pred_routes=vis_routes,
                        pred_scores=logits[image_index][
                            pred_mask if expert_index is not None
                            else slice(None)
                        ].sigmoid(),
                        gt_count=float(gt_count),
                        pred_count=sum(expert_counts),
                        title=f"{run_tag} | {row['filename']} | "
                        f"GT={gt_count} E0={expert_counts[0]:.1f} "
                        f"E1={expert_counts[1]:.1f} E2={expert_counts[2]:.1f}",
                        conf_threshold=args.vis_conf,
                    )
                    save_figure(
                        figure,
                        os.path.join(
                            vis_dir,
                            f"{batch_index:03d}_{image_index:02d}_"
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
                "count_metric": "native_sum_sigmoid (per expert)",
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
                )
    logging.info("汇总: %s", summary_path)
    logging.info("逐图明细: %s", csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单专家消融评估：逐 GT 最近距离、Recall@r、计数 MAE"
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
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    main(parse_args())
