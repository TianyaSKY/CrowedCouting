"""跨数据集分组评估：对多个数据集分别输出 MAE/RMSE。

复用 test_each_dataset.py 的计数口径（hard 路由 + 全部候选点 Σsigmoid）
与模型加载逻辑；按数据集各自汇总，另输出合并汇总表。

用法（在 GPU 机器上，从项目根目录）:

    python -m scripts.evaluation.evaluate_datasets \
        --checkpoint runs/moe_point/best.pt \
        --dataset shanghaitech=datasets/shanghaitech_AB:val \
        --dataset jhu=datasets/jhu_crowd:val \
        --dataset qnrf=datasets/ucf_qnrf:test \
        --dataset cc50_fold0=datasets/ucf_cc50:fold0_test \
        --out-dir runs/eval_datasets

UCF-CC-50 的 5 折需逐折指定（fold0_test..fold4_test），汇总时自行取均值。

输出 <out-dir>/:
    <name>/predictions.csv   逐图计数
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


def evaluate_datasets(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "evaluate.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    model, metadata = load_checkpoint_model(
        args.checkpoint, args.weights, device
    )
    logging.info(
        "checkpoint: epoch=%s best_mae=%s hidden=%s refs=%s",
        metadata["epoch"],
        metadata["best_mae"],
        metadata["hidden_channels"],
        metadata["num_references"],
    )

    specs = [parse_dataset_spec(s) for s in args.dataset]
    if not specs:
        raise SystemExit("至少需要一个 --dataset name=data_root:split")

    combined: dict[str, dict] = {}

    for name, root, split in specs:
        logging.info("== 评估数据集 %s (%s, split=%s) ==", name, root, split)

        dataset = PointDataset(
            root, split=split, crop_size=args.imgsz, augment=False
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
                        temperature=args.temperature,
                        hard_route=True,
                        router_grad=False,
                    )
                pred_counts = predictions["logits"].sigmoid().sum(dim=1)

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
            {"dataset": name, "root": root, "split": split}
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
                "imgsz": args.imgsz,
                "temperature": args.temperature,
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
        description="跨数据集分组评估（hard 路由 + Σsigmoid 计数）"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="MoE checkpoint 路径（runs/moe_point/best.pt）",
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11n.pt",
        help="checkpoint 未记录 weights 时的 fallback Backbone 权重",
    )
    parser.add_argument(
        "--dataset", type=str, action="append", default=[],
        metavar="NAME=ROOT:SPLIT",
        help="可多次指定；例如 shanghaitech=datasets/shanghaitech_AB:val",
    )
    parser.add_argument("--imgsz", type=int, default=640,
                        help="letterbox 尺寸，应与训练 crop_size 一致")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str,
                        default="runs/eval_datasets")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_datasets(parse_args())
