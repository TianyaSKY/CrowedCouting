"""Cross-dataset evaluation for the native_multiscale point head."""

import argparse
import csv
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data.point_dataset import PointDataset, point_collate_fn
from scripts.visualization.plot_utils import (
    create_moe_comparison_figure,
    generate_markdown_report,
    plot_benchmark_bar_chart,
    plot_count_scatter,
    save_figure,
)
from test_each_dataset import (
    CountMetrics,
    load_checkpoint_model,
    setup_logging,
)

NATIVE_ARCHITECTURE = "native_multiscale"


def parse_dataset_spec(spec: str) -> tuple[str, str, str]:
    name, _, rest = spec.partition("=")
    root, _, split = rest.partition(":")
    if not (name and root and split):
        raise SystemExit(
            f"--dataset 格式应为 name=data_root:split，收到: {spec}"
        )
    return name, root, split


def format_row(name: str, metrics: CountMetrics) -> str:
    values = metrics.as_dict()
    return (
        f"{name:<12} n={values['num_images']:>5} "
        f"MAE={values['mae']:8.3f} "
        f"RMSE={values['rmse']:8.3f} "
        f"NAE={values['nae']:.4f} "
        f"GT={values['gt_total']:>7} "
        f"Pred={values['pred_total']:8.1f}"
    )


def evaluate_datasets(args: argparse.Namespace) -> None:
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

    specs = [parse_dataset_spec(spec) for spec in args.dataset]
    if not specs:
        raise SystemExit("至少需要一个 --dataset name=data_root:split")

    combined: dict[str, dict] = {}
    dataset_details: list[dict] = []
    for name, root, split in specs:
        dataset = PointDataset(
            root,
            split=split,
            crop_size=crop_size,
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=point_collate_fn,
            pin_memory=device == "cuda",
        )
        metrics = CountMetrics()
        processed = 0
        detailed_records: list[dict[str, object]] = []
        out_ds = os.path.join(args.out_dir, name)
        os.makedirs(out_ds, exist_ok=True)
        csv_path = os.path.join(out_ds, "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["filename", "gt_count", "pred_count", "abs_error"]
            )
            with torch.inference_mode():
                for batch in tqdm(loader, desc=name):
                    images = batch["img"].to(device)
                    gt_points = batch["points"]
                    predictions = model(images)
                    pred_counts = (
                        predictions["logits"].sigmoid().sum(dim=1).cpu().tolist()
                    )
                    for index, (points, pred_count) in enumerate(
                        zip(gt_points, pred_counts)
                    ):
                        filename = os.path.basename(
                            dataset.image_paths[processed + index]
                        )
                        gt_count = int(points.shape[0])
                        pred_count = float(pred_count)
                        abs_error = abs(pred_count - gt_count)
                        metrics.update(gt_count, pred_count)
                        detailed_records.append(
                            {
                                "dataset_index": processed + index,
                                "filename": filename,
                                "gt_count": gt_count,
                                "pred_count": pred_count,
                                "abs_error": abs_error,
                            }
                        )
                        writer.writerow(
                            [
                                filename,
                                gt_count,
                                f"{pred_count:.3f}",
                                f"{abs_error:.3f}",
                            ]
                        )
                    processed += len(gt_points)

        summary = metrics.as_dict()
        summary.update(
            {
                "dataset": name,
                "root": root,
                "split": split,
                "architecture": NATIVE_ARCHITECTURE,
                "count_metric": "native_sum_sigmoid",
            }
        )
        with open(
            os.path.join(out_ds, "summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        combined[name] = summary

        if args.save_scatter and detailed_records:
            scatter = plot_count_scatter(
                [float(record["gt_count"]) for record in detailed_records],
                [float(record["pred_count"]) for record in detailed_records],
                metrics=summary,
                title=f"Dataset {name} ({split}): GT vs. Predicted Count",
            )
            save_figure(scatter, os.path.join(out_ds, "scatter.png"))

        if args.num_vis > 0 and detailed_records:
            vis_dir = os.path.join(out_ds, "eval_vis")
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
                    raw_sample = dataset[index]
                    sample_image = raw_sample["img"].unsqueeze(0).to(device)
                    sample_prediction = model(sample_image)
                    figure = create_moe_comparison_figure(
                        image=sample_image[0],
                        gt_points=raw_sample["points"],
                        pred_points=sample_prediction["points"][0].cpu(),
                        pred_routes=sample_prediction["expert_indices"][0].cpu(),
                        pred_scores=sample_prediction["logits"][0].sigmoid().cpu(),
                        gt_count=float(record["gt_count"]),
                        pred_count=float(record["pred_count"]),
                        title=(
                            f"[{tag.upper()}] {name}/{record['filename']} | "
                            f"GT: {record['gt_count']} | "
                            f"Pred: {float(record['pred_count']):.1f}"
                        ),
                    )
                    save_figure(
                        figure,
                        os.path.join(
                            vis_dir,
                            f"{tag}_{str(record['filename']).replace('.jpg', '')}.jpg",
                        ),
                    )

        dataset_details.append(
            {
                "name": name,
                "description": f"Root: {root} | Split: {split}",
                "samples": f"{summary['num_images']} images",
                "mae": summary["mae"],
                "rmse": summary["rmse"],
                "nae": summary["nae"],
                "gt_total": int(summary["gt_total"]),
                "pred_total": float(summary["pred_total"]),
                "artifacts": {"Predictions CSV": f"{name}/predictions.csv"},
            }
        )
        logging.info("  %s", format_row(name, metrics))

    overall_json = {
        "checkpoint": args.checkpoint,
        "architecture": NATIVE_ARCHITECTURE,
        "imgsz": crop_size,
        "datasets": combined,
        "checkpoint_metadata": metadata,
    }
    with open(
        os.path.join(args.out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(overall_json, file, indent=2, ensure_ascii=False)

    headers = [
        "Dataset",
        "Images (N)",
        "MAE",
        "RMSE",
        "NAE",
        "GT Total",
        "Pred Total",
        "Status",
    ]
    rows = []
    valid_maes = []
    valid_rmses = []
    valid_naes = []
    for name, summary in combined.items():
        valid_maes.append(float(summary["mae"]))
        valid_rmses.append(float(summary["rmse"]))
        valid_naes.append(float(summary["nae"]))
        rows.append(
            [
                name,
                str(summary["num_images"]),
                f"{float(summary['mae']):.3f}",
                f"{float(summary['rmse']):.3f}",
                f"{float(summary['nae']):.4f}",
                str(int(summary["gt_total"])),
                f"{float(summary['pred_total']):.1f}",
                "SUCCESS",
            ]
        )
    if valid_maes:
        rows.append(
            [
                "AVERAGE",
                "-",
                f"{np.mean(valid_maes):.3f}",
                f"{np.mean(valid_rmses):.3f}",
                f"{np.mean(valid_naes):.4f}",
                "-",
                "-",
                "-",
            ]
        )

    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    if args.report and combined:
        report_path = os.path.join(args.out_dir, "summary_report.md")
        generate_markdown_report(
            title="Native Multiscale Cross-Dataset Evaluation",
            meta={
                "Checkpoint": args.checkpoint,
                "Architecture": NATIVE_ARCHITECTURE,
                "Crop Size": crop_size,
                "Device": device,
            },
            headers=headers,
            rows=rows,
            dataset_details=dataset_details,
            output_path=report_path,
        )
        plot_benchmark_bar_chart(
            combined,
            os.path.join(args.out_dir, "benchmark_comparison.png"),
            title="Native Multiscale Cross-Dataset Performance",
        )

    logging.info("汇总 JSON: %s", os.path.join(args.out_dir, "summary.json"))
    logging.info("汇总 CSV: %s", summary_csv)


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估 native_multiscale P3/P4/P5 合并候选池"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        default=[],
        metavar="NAME=ROOT:SPLIT",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="runs/native_multiscale/eval_datasets")
    parser.add_argument("--save-scatter", action="store_true", default=True)
    parser.add_argument("--no-scatter", dest="save_scatter", action="store_false")
    parser.add_argument("--num-vis", type=int, default=6)
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--no-report", dest="report", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_datasets(parse_args())
