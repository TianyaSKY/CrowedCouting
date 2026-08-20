"""Evaluate YOLO11 + D2 MoE crowd counting on ShanghaiTech subsets.

The primary MAE uses deterministic Soft Top-2 routing, matching training
validation. ``--mode`` also exposes full3-soft and deterministic Top-1
diagnostics; all modes sum sigmoid foreground probabilities over candidates.

Example:
    python test_each_dataset.py \
        --data-root datasets/shanghaitech_AB \
        --checkpoint runs/moe_point/best_top2.pt \
        --mode top2 \
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


from dataclasses import dataclass

from scripts.visualization.plot_utils import (
    create_moe_comparison_figure,
    generate_markdown_report,
    plot_benchmark_bar_chart,
    plot_count_scatter,
    save_figure,
)


@dataclass
class CountMetrics:
    """Streaming count-error accumulator for one dataset subset."""

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
    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}

    def saved_value(name: str, default: object) -> object:
        if name in saved_config:
            return saved_config[name]
        return checkpoint_args.get(name, default)

    weights = str(
        checkpoint_args.get("weights", fallback_weights)
    )
    hidden_channels = int(
        checkpoint_args.get("hidden_channels", 128)
    )
    num_references = int(saved_value("num_references", 4))
    crop_size = int(saved_value("crop_size", 640))
    temperature_schedule = saved_config.get(
        "temperature_schedule",
        {
            name: checkpoint_args.get(name)
            for name in (
                "init_temperature",
                "phase1_temp",
                "soft_temp_floor",
                "temp_floor_epoch",
            )
            if name in checkpoint_args
        },
    )
    if not isinstance(temperature_schedule, dict):
        temperature_schedule = {}
    checkpoint_temperature = float(
        saved_config.get(
            "temperature",
            checkpoint.get(
                "temperature",
                temperature_schedule.get(
                    "soft_temp_floor",
                    1.0,
                ),
            ),
        )
    )
    router_active = bool(
        saved_value(
            "router_active",
            checkpoint.get("router_active", False),
        )
    )
    router_start_epoch = int(
        saved_value(
            "router_start_epoch",
            checkpoint_args.get("router_start_epoch", 6),
        )
    )
    router_training_mode = str(
        saved_value(
            "router_training_mode",
            checkpoint.get(
                "router_training_mode",
                "task_only_drop1_soft_top2",
            ),
        )
    )
    expert_dropout = str(
        saved_value(
            "expert_dropout",
            checkpoint.get("expert_dropout", "candidate_drop1"),
        )
    )
    active_experts = int(
        saved_value(
            "active_experts",
            checkpoint.get("active_experts", 2),
        )
    )
    scale_centers = saved_value(
        "scale_centers",
        [10.0, 20.0, 40.0],
    )
    if not isinstance(scale_centers, (list, tuple)):
        scale_centers = [10.0, 20.0, 40.0]

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
        "selection_metric": checkpoint.get("selection_metric"),
        "weights": weights,
        "hidden_channels": hidden_channels,
        "num_references": num_references,
        "crop_size": crop_size,
        "temperature": checkpoint_temperature,
        "temperature_schedule": temperature_schedule,
        "router_active": router_active,
        "router_start_epoch": router_start_epoch,
        "router_training_mode": router_training_mode,
        "expert_dropout": expert_dropout,
        "active_experts": active_experts,
        "scale_centers": [
            float(center) for center in scale_centers
        ],
        "config": saved_config,
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

    model, checkpoint_metadata = load_checkpoint_model(
        args.checkpoint,
        args.weights,
        device,
    )
    checkpoint_temperature = float(
        checkpoint_metadata["temperature"]
    )
    temperature = (
        checkpoint_temperature
        if args.temperature is None
        else args.temperature
    )

    crop_size = (
        args.imgsz
        if args.imgsz is not None
        else int(checkpoint_metadata["crop_size"])
    )
    logging.info(
        "checkpoint: epoch=%s best_mae=%s metric=%s hidden=%s refs=%s "
        "crop_size=%s temperature=%s mode=%s router_active=%s "
        "router_start_epoch=%s dropout=%s active_experts=%s",
        checkpoint_metadata["epoch"],
        checkpoint_metadata["best_mae"],
        checkpoint_metadata["selection_metric"],
        checkpoint_metadata["hidden_channels"],
        checkpoint_metadata["num_references"],
        crop_size,
        temperature,
        args.mode,
        checkpoint_metadata["router_active"],
        checkpoint_metadata["router_start_epoch"],
        checkpoint_metadata["expert_dropout"],
        checkpoint_metadata["active_experts"],
    )
    if not checkpoint_metadata["router_active"]:
        logging.warning(
            "当前 checkpoint 尚未启用 Router；仍按请求模式执行推理。"
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

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    detailed_records: list[dict[str, object]] = []
    processed_count = 0

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
                    temperature=temperature,
                    routing_mode=args.mode,
                )
                pred_counts = (
                    predictions["logits"]
                    .sigmoid()
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .tolist()
                )

                for i, (filename, subset, points, pred_count) in enumerate(
                    zip(filenames, subsets, gt_points, pred_counts)
                ):
                    gt_count = int(points.shape[0])
                    pred_count = float(pred_count)
                    abs_error = abs(pred_count - gt_count)

                    metrics["overall"].update(gt_count, pred_count)
                    metrics[str(subset)].update(gt_count, pred_count)

                    detailed_records.append(
                        {
                            "dataset_index": processed_count + i,
                            "filename": filename,
                            "subset": str(subset),
                            "gt_count": gt_count,
                            "pred_count": pred_count,
                            "abs_error": abs_error,
                        }
                    )

                    writer.writerow(
                        [
                            filename,
                            subset,
                            gt_count,
                            f"{pred_count:.6f}",
                            f"{abs_error:.6f}",
                        ]
                    )

                processed_count += len(gt_points)

    if metrics["overall"].num_images == 0:
        raise RuntimeError("没有成功评估任何图像")

    summary = {
        "count_metric": f"{args.mode}_sum_sigmoid",
        "temperature": temperature,
        "imgsz": crop_size,
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

    # 1. 生成并保存 GT vs Pred 回归散点图
    if args.save_scatter and detailed_records:
        overall_scatter = plot_count_scatter(
            [float(r["gt_count"]) for r in detailed_records],
            [float(r["pred_count"]) for r in detailed_records],
            metrics=metrics["overall"].as_dict(),
            title=f"ShanghaiTech ({args.split}): GT vs. Predicted Count",
        )
        save_figure(overall_scatter, os.path.join(args.out_dir, "overall_scatter.png"))
        logging.info(f"总览散点图已保存: {os.path.join(args.out_dir, 'overall_scatter.png')}")

        for subset_name in ("part_A", "part_B"):
            sub_records = [r for r in detailed_records if r["subset"] == subset_name]
            if sub_records:
                sub_scatter = plot_count_scatter(
                    [float(r["gt_count"]) for r in sub_records],
                    [float(r["pred_count"]) for r in sub_records],
                    metrics=metrics[subset_name].as_dict(),
                    title=f"ShanghaiTech {subset_name.replace('_', ' ').title()} ({args.split}): GT vs. Predicted",
                )
                save_figure(sub_scatter, os.path.join(args.out_dir, f"{subset_name}_scatter.png"))
                logging.info(f"{subset_name} 散点图已保存: {os.path.join(args.out_dir, f'{subset_name}_scatter.png')}")

    # 2. 生成定性样本对比图（难例 Worst-N 与优秀样本 Best-N）
    if args.num_vis > 0 and detailed_records:
        vis_dir = os.path.join(args.out_dir, "eval_vis")
        os.makedirs(vis_dir, exist_ok=True)
        sorted_records = sorted(
            detailed_records, key=lambda r: float(r["abs_error"]), reverse=True
        )
        half = max(1, args.num_vis // 2)
        selected: list[tuple[str, dict[str, object]]] = []
        for rank, r in enumerate(sorted_records[:half]):
            selected.append((f"worst_{rank + 1:02d}", r))
        for rank, r in enumerate(sorted_records[-half:]):
            selected.append((f"best_{rank + 1:02d}", r))

        with torch.inference_mode():
            for tag, record in selected:
                idx = int(record["dataset_index"])
                raw_sample = base_dataset[idx]
                sample_img = raw_sample["img"].unsqueeze(0).to(device)  # [1, 3, H, W]
                sample_gt = raw_sample["points"]
                sample_pred = model(
                    sample_img,
                    temperature=temperature,
                    routing_mode=args.mode,
                )
                p_scores = sample_pred["logits"][0].sigmoid().cpu()
                p_points = sample_pred["points"][0].cpu()
                p_routes = sample_pred["gates"][0].argmax(dim=-1).cpu()

                clean_id = str(record["filename"]).replace(".jpg", "")
                gt_c = float(record["gt_count"])
                pred_c = float(record["pred_count"])
                err_c = float(record["abs_error"])

                fig = create_moe_comparison_figure(
                    image=sample_img[0],
                    gt_points=sample_gt,
                    pred_points=p_points,
                    pred_routes=p_routes,
                    pred_scores=p_scores,
                    gt_count=gt_c,
                    pred_count=pred_c,
                    title=f"[{tag.upper()}] {clean_id} | GT: {gt_c:.0f} | Pred: {pred_c:.1f} (Err: {err_c:.1f})",
                )
                vis_path = os.path.join(
                    vis_dir, f"{tag}_{clean_id}_gt{gt_c:.0f}_pred{pred_c:.0f}.jpg"
                )
                save_figure(fig, vis_path)

        logging.info(f"定性对比图已保存 ({len(selected)} 张): {vis_dir}")

    # 3. 生成 Markdown 总结报告与 Benchmark 对比柱状图
    if args.report:
        bar_data = {
            name: metrics[name].as_dict()
            for name in ("part_A", "part_B")
            if metrics[name].num_images > 0
        }
        if bar_data:
            bar_chart_path = os.path.join(args.out_dir, "benchmark_comparison.png")
            plot_benchmark_bar_chart(
                bar_data,
                bar_chart_path,
                title=f"ShanghaiTech ({args.split}) Performance by Subset",
            )
            logging.info(f"子集对比柱状图已保存: {bar_chart_path}")

        headers = ["Subset", "Images (N)", "MAE", "RMSE", "NAE", "GT Total", "Pred Total"]
        rows = []
        details = []
        for s_name in ("part_A", "part_B", "overall"):
            if metrics[s_name].num_images > 0:
                m = metrics[s_name].as_dict()
                display = "Part A" if s_name == "part_A" else ("Part B" if s_name == "part_B" else "Overall")
                rows.append([
                    display,
                    str(m["num_images"]),
                    f"{m['mae']:.3f}",
                    f"{m['rmse']:.3f}",
                    f"{m['nae']:.4f}",
                    str(int(m["gt_total"])),
                    f"{float(m['pred_total']):.1f}",
                ])
                scatter_f = f"{s_name}_scatter.png"
                arts = {
                    "Predictions CSV": "predictions.csv",
                    "Scatter Plot": scatter_f if os.path.exists(os.path.join(args.out_dir, scatter_f)) else "",
                }
                details.append({
                    "name": f"ShanghaiTech {display}",
                    "samples": f"{m['num_images']} images",
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "nae": m["nae"],
                    "gt_total": int(m["gt_total"]),
                    "pred_total": float(m["pred_total"]),
                    "artifacts": {k: v for k, v in arts.items() if v},
                })

        report_path = os.path.join(args.out_dir, "summary_report.md")
        generate_markdown_report(
            title="YOLO11-MoE 人群计数评测报告 (ShanghaiTech Evaluation Report)",
            meta={
                "Checkpoint": args.checkpoint,
                "Split": args.split,
                "Routing Mode": args.mode,
                "Crop Size": crop_size,
                "Temperature": temperature,
                "Device": device,
            },
            headers=headers,
            rows=rows,
            dataset_details=details,
            output_path=report_path,
        )
        logging.info(f"基准总结报告已生成: {report_path}")

    logging.info("=" * 72)
    logging.info(
        "MoE 测试结果（%s + sum(sigmoid)）",
        args.mode,
    )
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
        description=(
            "MoE 人群计数测试：full3-soft / deterministic Top-2 / Top-1"
        )
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
        help="评估 split（训练期默认 val；官方 test 应在训练结束后指定）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/moe_point/best_top2.pt",
        help="MoE checkpoint，默认使用 best_top2.pt",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11m.pt",
        help="旧 checkpoint 未记录 weights 时使用的 fallback Backbone 权重",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="覆盖 checkpoint crop_size；默认读取 checkpoint 配置",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="覆盖推理温度；默认读取 checkpoint 保存的温度",
    )
    parser.add_argument(
        "--mode",
        choices=("full3_soft", "top2", "top1"),
        default="top2",
        help="确定性评估模式；默认 D2 主指标 Top-2",
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
    parser.add_argument(
        "--save-scatter",
        action="store_true",
        default=True,
        help="保存 GT vs Pred 回归散点图（默认开启）",
    )
    parser.add_argument(
        "--no-scatter",
        dest="save_scatter",
        action="store_false",
        help="禁用散点图生成",
    )
    parser.add_argument(
        "--num-vis",
        type=int,
        default=6,
        help="保存定性样本对比图数量（默认 6，兼顾最大误差与最小误差样本）",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="生成 summary_report.md 总结报告与对比柱状图（默认开启）",
    )
    parser.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="禁用 Markdown 报告生成",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

