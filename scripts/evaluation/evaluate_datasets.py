"""跨数据集分组评估：对多个数据集分别输出 MAE/RMSE。

默认使用 D2 deterministic Soft Top-2；``--mode`` 可切换 full3-soft 或
deterministic Top-1。三种模式都采用全部候选点 ``Σsigmoid`` 计数口径。

用法（在 GPU 机器上，从项目根目录）:

    python -m scripts.evaluation.evaluate_datasets \
        --checkpoint runs/moe_point_all/best_top2.pt \
        --mode top2 \
        --dataset shanghaitech=datasets/shanghaitech_AB:val \
        --dataset jhu=datasets/jhu_crowd:val \
        --dataset qnrf=datasets/ucf_qnrf:test \
        --dataset cc50_fold0=datasets/ucf_cc50:fold0_test \
        --out-dir runs/eval_datasets

UCF-CC-50 的 5 折需逐折指定（fold0_test..fold4_test），汇总时自行取均值。

输出 <out-dir>/:
    <name>/predictions.csv  逐图计数
    <name>/summary.json      该数据集 MAE/RMSE/GT/Pred 汇总
    summary.json             全部数据集合并汇总
    evaluate.log             控制台日志副本
"""

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
from test_each_dataset import CountMetrics, load_checkpoint_model, setup_logging


def parse_dataset_spec(spec: str) -> tuple[str, str, str]:
    """解析 'name=data_root:split'。"""
    name, _, rest = spec.partition("=")
    root, _, split = rest.partition(":")
    if not (name and root and split):
        raise SystemExit(
            f"--dataset 格式应为 name=data_root:split，收到: {spec}"
        )
    return name, root, split


def format_row(name: str, metrics: CountMetrics) -> str:
    """格式化单个数据集的汇总日志。"""
    values = metrics.as_dict()
    nae_str = f"NAE={values.get('nae', 0.0):.4f}" if "nae" in values else ""
    return (
        f"{name:<12} n={values['num_images']:>5} "
        f"MAE={values['mae']:8.3f} "
        f"RMSE={values['rmse']:8.3f} "
        f"{nae_str} "
        f"GT={values['gt_total']:>7} "
        f"Pred={values['pred_total']:8.1f}"
    )


def evaluate_datasets(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "evaluate.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    model, metadata = load_checkpoint_model(
        args.checkpoint, args.weights, device
    )
    checkpoint_temperature = float(
        metadata["temperature"]
    )
    temperature = (
        checkpoint_temperature
        if args.temperature is None
        else args.temperature
    )
    crop_size = (
        args.imgsz
        if args.imgsz is not None
        else int(metadata["crop_size"])
    )
    logging.info(
        "checkpoint: epoch=%s best_mae=%s metric=%s hidden=%s refs=%s "
        "crop_size=%s temperature=%s mode=%s router_active=%s "
        "router_start_epoch=%s dropout=%s active_experts=%s",
        metadata["epoch"],
        metadata["best_mae"],
        metadata["selection_metric"],
        metadata["hidden_channels"],
        metadata["num_references"],
        crop_size,
        temperature,
        args.mode,
        metadata["router_active"],
        metadata["router_start_epoch"],
        metadata["expert_dropout"],
        metadata["active_experts"],
    )
    if not metadata["router_active"]:
        logging.warning(
            "当前 checkpoint 尚未启用 Router；仍按请求模式执行推理。"
        )

    specs = [parse_dataset_spec(s) for s in args.dataset]
    if not specs:
        raise SystemExit("至少需要一个 --dataset name=data_root:split")

    combined: dict[str, dict] = {}
    dataset_details: list[dict] = []

    for name, root, split in specs:
        dataset = PointDataset(
            root, split=split, crop_size=crop_size, augment=False
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
        out_ds = os.path.join(args.out_dir, name)
        os.makedirs(out_ds, exist_ok=True)

        detailed_records: list[dict[str, object]] = []

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
                    predictions = model(
                        images,
                        temperature=temperature,
                        routing_mode=args.mode,
                    )
                    pred_counts = predictions["logits"].sigmoid().sum(
                        dim=1
                    )

                    for i in range(len(gt_points)):
                        filename = os.path.basename(
                            dataset.image_paths[processed + i]
                        )
                        gt_count = int(gt_points[i].shape[0])
                        pred_count = float(pred_counts[i].item())
                        abs_error = abs(pred_count - gt_count)

                        metrics.update(gt_count, pred_count)
                        detailed_records.append(
                            {
                                "dataset_index": processed + i,
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
                "mode": args.mode,
                "count_metric": f"{args.mode}_sum_sigmoid",
            }
        )
        with open(
            os.path.join(out_ds, "summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        combined[name] = summary

        # 1. 散点图
        scatter_rel_path = ""
        if args.save_scatter and detailed_records:
            scatter_fig = plot_count_scatter(
                [float(r["gt_count"]) for r in detailed_records],
                [float(r["pred_count"]) for r in detailed_records],
                metrics=summary,
                title=f"Dataset {name} ({split}): GT vs. Predicted Count",
            )
            scatter_path = os.path.join(out_ds, "scatter.png")
            save_figure(scatter_fig, scatter_path)
            scatter_rel_path = f"{name}/scatter.png"
            logging.info("  %s 散点图已保存: %s", name, scatter_path)

        # 2. 定性样本对比图
        if args.num_vis > 0 and detailed_records:
            vis_dir = os.path.join(out_ds, "eval_vis")
            os.makedirs(vis_dir, exist_ok=True)
            sorted_records = sorted(
                detailed_records,
                key=lambda r: float(r["abs_error"]),
                reverse=True,
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
                    raw_sample = dataset[idx]
                    sample_img = raw_sample["img"].unsqueeze(0).to(device)
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
                        title=f"[{tag.upper()}] {name}/{clean_id} | GT: {gt_c:.0f} | Pred: {pred_c:.1f} (Err: {err_c:.1f})",
                    )
                    vis_path = os.path.join(
                        vis_dir,
                        f"{tag}_{clean_id}_gt{gt_c:.0f}_pred{pred_c:.0f}.jpg",
                    )
                    save_figure(fig, vis_path)

            logging.info(
                "  %s 定性对比图已保存 (%d 张): %s",
                name,
                len(selected),
                vis_dir,
            )

        arts = {
            "Predictions CSV": f"{name}/predictions.csv",
        }
        if scatter_rel_path:
            arts["Scatter Plot"] = scatter_rel_path
        if args.num_vis > 0:
            arts["Sample Visualizations"] = f"{name}/eval_vis/"

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
                "artifacts": arts,
            }
        )

        logging.info("  %s", format_row(name, metrics))

    # 跨数据集整体产物
    overall_json = {
        "checkpoint": args.checkpoint,
        "imgsz": crop_size,
        "temperature": temperature,
        "checkpoint_metadata": metadata,
        "datasets": combined,
    }
    with open(
        os.path.join(args.out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall_json, f, indent=2, ensure_ascii=False)

    # 3. 生成跨数据集对比柱状图
    if args.report and combined:
        bar_chart_path = os.path.join(
            args.out_dir, "benchmark_comparison.png"
        )
        plot_benchmark_bar_chart(
            combined,
            bar_chart_path,
            title="Crowd Counting Cross-Dataset Benchmark Performance",
        )
        logging.info("基准对比柱状图已保存: %s", bar_chart_path)

    # 4. 生成 CSV 结构化汇总表
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

    for name, d in combined.items():
        v_mae = float(d["mae"])
        v_rmse = float(d["rmse"])
        v_nae = float(d["nae"])
        valid_maes.append(v_mae)
        valid_rmses.append(v_rmse)
        valid_naes.append(v_nae)

        rows.append(
            [
                name,
                str(d["num_images"]),
                f"{v_mae:.3f}",
                f"{v_rmse:.3f}",
                f"{v_nae:.4f}",
                str(int(d["gt_total"])),
                f"{float(d['pred_total']):.1f}",
                "SUCCESS",
            ]
        )

    if valid_maes:
        rows.append(
            [
                "★ AVERAGE",
                "-",
                f"{np.mean(valid_maes):.3f}",
                f"{np.mean(valid_rmses):.3f}",
                f"{np.mean(valid_naes):.4f}",
                "-",
                "-",
                "-",
            ]
        )

    csv_summary_path = os.path.join(args.out_dir, "summary_metrics.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # 5. 生成 Markdown 总结报告
    if args.report:
        report_path = os.path.join(args.out_dir, "summary_report.md")
        generate_markdown_report(
            title="YOLO11-MoE 跨数据集人群计数基准评测报告 (Benchmark Report)",
            meta={
                "Checkpoint": args.checkpoint,
                "Routing Mode": args.mode,
                "Crop Size": crop_size,
                "Temperature": temperature,
                "Device": device,
            },
            headers=headers,
            rows=rows,
            dataset_details=dataset_details,
            output_path=report_path,
        )
        logging.info("基准评测 Markdown 报告已生成: %s", report_path)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS:")
    print("-" * 80)
    fmt = "{:<16}{:>10}{:>10}{:>10}{:>10}{:>10}{:>12}"
    print(
        fmt.format(
            "Dataset", "Images", "MAE", "RMSE", "NAE", "GT Total", "Pred Total"
        )
    )
    print("-" * 80)
    for name, d in combined.items():
        print(
            fmt.format(
                name,
                d["num_images"],
                f"{d['mae']:.3f}",
                f"{d['rmse']:.3f}",
                f"{d['nae']:.4f}",
                d["gt_total"],
                f"{d['pred_total']:.1f}",
            )
        )
    if valid_maes:
        print("-" * 80)
        print(
            fmt.format(
                "★ AVERAGE",
                "-",
                f"{np.mean(valid_maes):.3f}",
                f"{np.mean(valid_rmses):.3f}",
                f"{np.mean(valid_naes):.4f}",
                "-",
                "-",
            )
        )
    print("=" * 80)
    print(f"汇总 JSON:   {os.path.join(args.out_dir, 'summary.json')}")
    print(f"汇总 CSV:    {csv_summary_path}")
    if args.report:
        print(f"评测报告 MD: {os.path.join(args.out_dir, 'summary_report.md')}")
        print(f"对比柱状图:  {os.path.join(args.out_dir, 'benchmark_comparison.png')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "跨数据集分组评估（full3-soft / deterministic Top-2 / Top-1）"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="MoE checkpoint 路径（默认正式使用 best_top2.pt）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11m.pt",
        help="checkpoint 未记录 weights 时的 fallback Backbone 权重",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        default=[],
        metavar="NAME=ROOT:SPLIT",
        help="可多次指定；例如 shanghaitech=datasets/shanghaitech_AB:val",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/eval_datasets",
    )
    parser.add_argument(
        "--save-scatter",
        action="store_true",
        default=True,
        help="保存每个数据集的 GT vs Pred 回归散点图（默认开启）",
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
        help="每个数据集保存定性样本对比图数量（默认 6，兼顾最大误差与最小误差样本）",
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
    evaluate_datasets(parse_args())
