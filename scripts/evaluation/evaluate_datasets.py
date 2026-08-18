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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data.point_dataset import PointDataset, point_collate_fn
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
    return (
        f"{name:<12} n={values['num_images']:>5} "
        f"MAE={values['mae']:8.3f} "
        f"RMSE={values['rmse']:8.3f} "
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

        csv_path = os.path.join(out_ds, "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["filename", "gt_count", "pred_count", "abs_error"]
            )

            for batch in tqdm(loader, desc=name):
                images = batch["img"].to(device)
                gt_points = batch["points"]
                with torch.no_grad():
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
                    metrics.update(gt_count, pred_count)
                    writer.writerow(
                        [
                            filename,
                            gt_count,
                            f"{pred_count:.3f}",
                            f"{abs(pred_count - gt_count):.3f}",
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
        logging.info("  %s", format_row(name, metrics))

    with open(
        os.path.join(args.out_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "imgsz": crop_size,
                "temperature": temperature,
                "checkpoint_metadata": metadata,
                "datasets": combined,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== 跨数据集汇总 =====")
    for name, d in combined.items():
        print(
            f"{name:<12} n={d['num_images']:>5}  "
            f"MAE={d['mae']:8.3f}  RMSE={d['rmse']:8.3f}  "
            f"GT={d['gt_total']:>7}  Pred={d['pred_total']:8.1f}"
        )
    print(f"汇总 JSON: {os.path.join(args.out_dir, 'summary.json')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "跨数据集分组评估（full3-soft / deterministic Top-2 / Top-1）"
        )
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="MoE checkpoint 路径（默认正式使用 best_top2.pt）",
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11m.pt",
        help="checkpoint 未记录 weights 时的 fallback Backbone 权重",
    )
    parser.add_argument(
        "--dataset", type=str, action="append", default=[],
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
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_datasets(parse_args())
