"""Evaluate YOLO11 + Scale-MoE crowd counting on ShanghaiTech subsets.

The primary MAE uses the same counting definition as training validation:
run the model with hard routing, then sum sigmoid foreground probabilities over
all candidates for each image. Results are reported for Part A, Part B, and the
whole requested split.

Example:
    python test_each_dataset.py \
        --data-root datasets/shanghaitech_AB \
        --checkpoint runs/moe_point/best.pt \
        --split val \
        --batch-size 16 \
        --out-dir runs/moe_point/test_eval

Outputs:
    <out-dir>/predictions.csv  per-image counts and absolute errors
    <out-dir>/summary.json     Part A / Part B / overall MAE and RMSE
    <out-dir>/evaluate.log     console-equivalent evaluation log
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.yolo11_moe_point import YOLO11MoEPoint
from scripts.data.point_dataset import PointDataset, point_collate_fn


@dataclass
class CountMetrics:
    """Streaming count-error accumulator for one dataset subset."""

    num_images: int = 0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    gt_total: int = 0
    pred_total: float = 0.0

    def update(self, gt_count: int, pred_count: float) -> None:
        error = pred_count - gt_count
        self.num_images += 1
        self.abs_error_sum += abs(error)
        self.squared_error_sum += error * error
        self.gt_total += gt_count
        self.pred_total += pred_count

    def as_dict(self) -> dict[str, float | int]:
        if self.num_images == 0:
            return {
                "num_images": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "gt_total": 0,
                "pred_total": 0.0,
            }

        return {
            "num_images": self.num_images,
            "mae": self.abs_error_sum / self.num_images,
            "rmse": (self.squared_error_sum / self.num_images) ** 0.5,
            "gt_total": self.gt_total,
            "pred_total": self.pred_total,
        }


class NamedPointDataset(Dataset):
    """Attach filename/subset metadata without changing the training dataset."""

    def __init__(self, dataset: PointDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.dataset[index]
        filename = os.path.basename(self.dataset.image_paths[index])
        base_name = os.path.splitext(filename)[0]
        return {
            **sample,
            "filename": filename,
            "subset": infer_subset(base_name),
        }


def infer_subset(base_name: str) -> str:
    """Infer ShanghaiTech Part A / Part B from prepare_combined prefixes."""
    if base_name.startswith("part_A_"):
        return "part_A"
    if base_name.startswith("part_B_"):
        return "part_B"
    return "other"


def evaluation_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Reuse the training collate function and preserve evaluation metadata."""
    model_batch = point_collate_fn(batch)  # type: ignore[arg-type]
    model_batch["filenames"] = [sample["filename"] for sample in batch]
    model_batch["subsets"] = [sample["subset"] for sample in batch]
    return model_batch


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


def load_checkpoint_model(
    checkpoint_path: str,
    fallback_weights: str,
    device: str,
) -> tuple[YOLO11MoEPoint, dict[str, object]]:
    """Build the exact MoE shape recorded in the checkpoint and load weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("MoE checkpoint 必须是包含 model state_dict 的字典")

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    weights = str(checkpoint_args.get("weights", fallback_weights))
    hidden_channels = int(checkpoint_args.get("hidden_channels", 128))
    num_references = int(checkpoint_args.get("num_references", 4))

    model = YOLO11MoEPoint(
        weights=weights,
        hidden_channels=hidden_channels,
        num_references=num_references,
    ).to(device)

    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise KeyError("checkpoint 中缺少 model state_dict")

    model.load_state_dict(state_dict)
    model.eval()

    metadata: dict[str, object] = {
        "epoch": checkpoint.get("epoch"),
        "best_mae": checkpoint.get("best_mae"),
        "weights": weights,
        "hidden_channels": hidden_channels,
        "num_references": num_references,
    }
    return model, metadata


def format_metrics(name: str, metrics: CountMetrics) -> str:
    values = metrics.as_dict()
    return (
        f"{name:<8} n={values['num_images']:>3} "
        f"MAE={values['mae']:.3f} "
        f"RMSE={values['rmse']:.3f} "
        f"GT={values['gt_total']} "
        f"Pred={values['pred_total']:.1f}"
    )


def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "evaluate.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    model, checkpoint_metadata = load_checkpoint_model(
        args.checkpoint,
        args.weights,
        device,
    )
    logging.info(
        "checkpoint: epoch=%s best_mae=%s hidden=%s refs=%s",
        checkpoint_metadata["epoch"],
        checkpoint_metadata["best_mae"],
        checkpoint_metadata["hidden_channels"],
        checkpoint_metadata["num_references"],
    )

    base_dataset = PointDataset(
        args.data_root,
        split=args.split,
        crop_size=args.imgsz,
        augment=False,
    )
    dataset = NamedPointDataset(base_dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=evaluation_collate,
        pin_memory=device == "cuda",
    )

    metrics = {
        "overall": CountMetrics(),
        "part_A": CountMetrics(),
        "part_B": CountMetrics(),
        "other": CountMetrics(),
    }

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "filename",
                "subset",
                "gt_count",
                "pred_count",
                "abs_error",
            ]
        )

        with torch.inference_mode():
            for batch in tqdm(loader, desc="测试中", leave=False):
                images = batch["img"].to(  # type: ignore[union-attr]
                    device,
                    non_blocking=True,
                )
                gt_points = batch["points"]
                filenames = batch["filenames"]
                subsets = batch["subsets"]

                predictions = model(
                    images,
                    temperature=args.temperature,
                    hard_route=True,
                )
                pred_counts = (
                    predictions["logits"]
                    .sigmoid()
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )

                for filename, subset, points, pred_count in zip(
                    filenames,
                    subsets,
                    gt_points,
                    pred_counts,
                ):
                    gt_count = int(points.shape[0])
                    pred_count = float(pred_count)
                    abs_error = abs(pred_count - gt_count)

                    metrics["overall"].update(gt_count, pred_count)
                    metrics[str(subset)].update(gt_count, pred_count)

                    writer.writerow(
                        [
                            filename,
                            subset,
                            gt_count,
                            f"{pred_count:.6f}",
                            f"{abs_error:.6f}",
                        ]
                    )

    if metrics["overall"].num_images == 0:
        raise RuntimeError("没有成功评估任何图像")

    summary = {
        "split": args.split,
        "count_metric": "hard_route_sum_sigmoid",
        "temperature": args.temperature,
        "imgsz": args.imgsz,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata,
        "overall": metrics["overall"].as_dict(),
        "subsets": {
            name: metrics[name].as_dict()
            for name in ("part_A", "part_B", "other")
            if metrics[name].num_images > 0
        },
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    logging.info("=" * 72)
    logging.info("MoE 测试结果（hard route + sum(sigmoid)，与训练 hard_MAE 一致）")
    for subset_name, display_name in (
        ("part_A", "Part A"),
        ("part_B", "Part B"),
        ("other", "Other"),
    ):
        if metrics[subset_name].num_images > 0:
            logging.info(format_metrics(display_name, metrics[subset_name]))
    logging.info(format_metrics("Overall", metrics["overall"]))
    logging.info("=" * 72)
    logging.info("逐图结果: %s", csv_path)
    logging.info("汇总结果: %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE 人群计数测试：分别输出 ShanghaiTech Part A/B MAE"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/shanghaitech_AB",
        help="数据集根目录（含 images/ 与 points/）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="评估 split。prepare_combined.py 将原始 test_data 映射为 val",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/moe_point/best.pt",
        help="MoE checkpoint，测试应优先使用 best.pt",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="旧 checkpoint 未记录 weights 时使用的 fallback Backbone 权重",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="测试 letterbox 尺寸，应与训练 crop_size 一致",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="hard route 的 softmax 温度；默认与训练后期验证一致",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/moe_point/test_eval",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())