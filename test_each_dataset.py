"""Evaluate the native_multiscale point head on ShanghaiTech subsets."""
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
from scripts.visualization.plot_utils import (
    create_moe_comparison_figure,
    generate_markdown_report,
    plot_benchmark_bar_chart,
    plot_count_scatter,
    save_figure,
)

NATIVE_ARCHITECTURE = "native_multiscale"


@dataclass
class CountMetrics:
    num_images: int = 0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    nae_sum: float = 0.0
    gt_total: int = 0
    pred_total: float = 0.0

    def update(self, gt_count: int, pred_count: float) -> None:
        error = pred_count - gt_count
        self.num_images += 1
        self.abs_error_sum += abs(error)
        self.squared_error_sum += error * error
        self.nae_sum += abs(error) / max(float(gt_count), 1.0)
        self.gt_total += gt_count
        self.pred_total += pred_count

    def as_dict(self) -> dict[str, float | int]:
        if self.num_images == 0:
            return {
                "num_images": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "nae": 0.0,
                "gt_total": 0,
                "pred_total": 0.0,
            }
        return {
            "num_images": self.num_images,
            "mae": self.abs_error_sum / self.num_images,
            "rmse": (self.squared_error_sum / self.num_images) ** 0.5,
            "nae": self.nae_sum / self.num_images,
            "gt_total": self.gt_total,
            "pred_total": self.pred_total,
        }


class NamedPointDataset(Dataset):
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
    if base_name.startswith("part_A_"):
        return "part_A"
    if base_name.startswith("part_B_"):
        return "part_B"
    return "other"


def evaluation_collate(batch: list[dict[str, object]]) -> dict[str, object]:
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


def _native_checkpoint_settings(
    checkpoint: dict[str, object],
    fallback_weights: str,
) -> tuple[str, int, tuple[int, int, int], int]:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or config.get("architecture") != NATIVE_ARCHITECTURE:
        raise ValueError(
            "只支持 native_multiscale checkpoint；旧 D2 checkpoint 已删除"
        )
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    weights = str(checkpoint_args.get("weights", fallback_weights))
    hidden_channels = int(
        config.get(
            "hidden_channels",
            checkpoint_args.get("hidden_channels", 256),
        )
    )
    raw_references = config.get(
        "native_references",
        checkpoint_args.get("native_references", (1, 4, 16)),
    )
    if isinstance(raw_references, str):
        references = tuple(
            int(item.strip()) for item in raw_references.split(",")
        )
    else:
        references = tuple(int(item) for item in raw_references)
    if len(references) != 3:
        raise ValueError("checkpoint native_references 必须包含三个值")
    crop_size = int(config.get("crop_size", checkpoint_args.get("crop_size", 640)))
    return weights, hidden_channels, (references[0], references[1], references[2]), crop_size


def load_checkpoint_model(
    checkpoint_path: str,
    fallback_weights: str,
    device: str,
) -> tuple[YOLO11MoEPoint, dict[str, object]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是包含 model state_dict 的字典")

    weights, hidden_channels, native_references, crop_size = (
        _native_checkpoint_settings(checkpoint, fallback_weights)
    )
    model = YOLO11MoEPoint(
        weights=weights,
        hidden_channels=hidden_channels,
        native_references=native_references,
    ).to(device)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise KeyError("checkpoint 中缺少 model state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    config = checkpoint["config"]
    assert isinstance(config, dict)
    metadata: dict[str, object] = {
        "architecture": NATIVE_ARCHITECTURE,
        "epoch": checkpoint.get("epoch"),
        "best_mae": checkpoint.get("best_mae"),
        "selection_metric": checkpoint.get("selection_metric"),
        "weights": weights,
        "hidden_channels": hidden_channels,
        "native_references": native_references,
        "crop_size": crop_size,
        "native_warmup_epochs": config.get("native_warmup_epochs"),
        "matching_stage": config.get("matching_stage"),
        "config": config,
    }
    return model, metadata


def format_metrics(name: str, metrics: CountMetrics) -> str:
    values = metrics.as_dict()
    return (
        f"{name:<8} n={values['num_images']:>3} "
        f"MAE={values['mae']:.3f} "
        f"RMSE={values['rmse']:.3f} "
        f"NAE={values['nae']:.4f} "
        f"GT={values['gt_total']} "
        f"Pred={values['pred_total']:.1f}"
    )


def evaluate(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "evaluate.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    model, metadata = load_checkpoint_model(
        args.checkpoint,
        args.weights,
        device,
    )
    crop_size = args.imgsz or int(metadata["crop_size"])
    logging.info(
        "checkpoint: epoch=%s best_mae=%s hidden=%s refs=%s crop_size=%s architecture=%s",
        metadata["epoch"],
        metadata["best_mae"],
        metadata["hidden_channels"],
        metadata["native_references"],
        crop_size,
        metadata["architecture"],
    )

    base_dataset = PointDataset(
        args.data_root,
        split=args.split,
        crop_size=crop_size,
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
    detailed_records: list[dict[str, object]] = []
    processed_count = 0
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["filename", "subset", "gt_count", "pred_count", "abs_error"]
        )
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Native 测试中", leave=False):
                images = batch["img"].to(device, non_blocking=True)
                gt_points = batch["points"]
                filenames = batch["filenames"]
                subsets = batch["subsets"]
                predictions = model(images)
                pred_counts = (
                    predictions["logits"]
                    .sigmoid()
                    .sum(dim=1)
                    .cpu()
                    .tolist()
                )
                for index, (filename, subset, points, pred_count) in enumerate(
                    zip(filenames, subsets, gt_points, pred_counts)
                ):
                    gt_count = int(points.shape[0])
                    pred_count = float(pred_count)
                    abs_error = abs(pred_count - gt_count)
                    metrics["overall"].update(gt_count, pred_count)
                    metrics[str(subset)].update(gt_count, pred_count)
                    detailed_records.append(
                        {
                            "dataset_index": processed_count + index,
                            "filename": filename,
                            "subset": str(subset),
                            "gt_count": gt_count,
                            "pred_count": pred_count,
                            "abs_error": abs_error,
                        }
                    )
                    writer.writerow(
                        [filename, subset, gt_count, f"{pred_count:.6f}", f"{abs_error:.6f}"]
                    )
                processed_count += len(gt_points)

    if metrics["overall"].num_images == 0:
        raise RuntimeError("没有成功评估任何图像")

    summary = {
        "count_metric": "native_sum_sigmoid",
        "imgsz": crop_size,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": metadata,
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

    if args.save_scatter and detailed_records:
        overall = metrics["overall"].as_dict()
        figure = plot_count_scatter(
            [float(record["gt_count"]) for record in detailed_records],
            [float(record["pred_count"]) for record in detailed_records],
            metrics=overall,
            title=f"ShanghaiTech ({args.split}): GT vs. Predicted Count",
        )
        save_figure(figure, os.path.join(args.out_dir, "overall_scatter.png"))

    if args.num_vis > 0 and detailed_records:
        vis_dir = os.path.join(args.out_dir, "eval_vis")
        os.makedirs(vis_dir, exist_ok=True)
        sorted_records = sorted(
            detailed_records,
            key=lambda record: float(record["abs_error"]),
            reverse=True,
        )
        half = max(1, args.num_vis // 2)
        selected = [
            (f"worst_{rank + 1:02d}", record)
            for rank, record in enumerate(sorted_records[:half])
        ] + [
            (f"best_{rank + 1:02d}", record)
            for rank, record in enumerate(sorted_records[-half:])
        ]
        with torch.inference_mode():
            for tag, record in selected:
                index = int(record["dataset_index"])
                raw_sample = base_dataset[index]
                sample_image = raw_sample["img"].unsqueeze(0).to(device)
                sample_prediction = model(sample_image)
                sample_scores = sample_prediction["logits"][0].sigmoid().cpu()
                sample_points = sample_prediction["points"][0].cpu()
                sample_sources = sample_prediction["expert_indices"][0].cpu()
                clean_id = str(record["filename"]).replace(".jpg", "")
                figure = create_moe_comparison_figure(
                    image=sample_image[0],
                    gt_points=raw_sample["points"],
                    pred_points=sample_points,
                    pred_routes=sample_sources,
                    pred_scores=sample_scores,
                    gt_count=float(record["gt_count"]),
                    pred_count=float(record["pred_count"]),
                    title=(
                        f"[{tag.upper()}] {clean_id} | "
                        f"GT: {record['gt_count']} | "
                        f"Pred: {float(record['pred_count']):.1f}"
                    ),
                )
                save_figure(
                    figure,
                    os.path.join(
                        vis_dir,
                        f"{tag}_{clean_id}_gt{int(record['gt_count'])}_pred{int(record['pred_count'])}.jpg",
                    ),
                )

    if args.report:
        rows = []
        for name in ("part_A", "part_B", "other"):
            if metrics[name].num_images == 0:
                continue
            values = metrics[name].as_dict()
            rows.append(
                [
                    name,
                    str(values["num_images"]),
                    f"{values['mae']:.3f}",
                    f"{values['rmse']:.3f}",
                    f"{values['nae']:.4f}",
                    str(values["gt_total"]),
                    f"{values['pred_total']:.1f}",
                    "SUCCESS",
                ]
            )
        if rows:
            report_path = os.path.join(args.out_dir, "summary_report.md")
            generate_markdown_report(
                title="Native Multiscale ShanghaiTech Evaluation",
                meta={
                    "Checkpoint": args.checkpoint,
                    "Architecture": NATIVE_ARCHITECTURE,
                    "Crop Size": crop_size,
                    "Device": device,
                },
                headers=[
                    "Dataset",
                    "Images (N)",
                    "MAE",
                    "RMSE",
                    "NAE",
                    "GT Total",
                    "Pred Total",
                    "Status",
                ],
                rows=rows,
                dataset_details=[],
                output_path=report_path,
            )
            plot_benchmark_bar_chart(
                {
                    name: metrics[name].as_dict()
                    for name in ("part_A", "part_B", "other")
                    if metrics[name].num_images > 0
                },
                os.path.join(args.out_dir, "benchmark_comparison.png"),
                title="Native Multiscale ShanghaiTech Performance",
            )

    logging.info("%s", format_metrics("Overall", metrics["overall"]))
    for name in ("part_A", "part_B", "other"):
        if metrics[name].num_images > 0:
            logging.info("%s", format_metrics(name, metrics[name]))
    logging.info("逐图结果: %s", csv_path)
    logging.info("汇总结果: %s", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 native_multiscale P3/P4/P5 合并候选池"
    )
    parser.add_argument("--data-root", type=str, default="datasets/shanghaitech_AB")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/native_multiscale/best_native.pt",
    )
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="runs/native_multiscale/test_eval")
    parser.add_argument("--save-scatter", action="store_true", default=True)
    parser.add_argument("--no-scatter", dest="save_scatter", action="store_false")
    parser.add_argument("--num-vis", type=int, default=6)
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--no-report", dest="report", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
