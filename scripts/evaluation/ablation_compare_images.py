"""Per-image cross-run comparison panels: E0-only / E1-only / E2-only / joint.

对每张 val 图生成一张 2x2 对比图，供人工横向检查“大目标”检出差异：
- 左上: E0-only (P3)      - 右上: E1-only (P4)
- 左下: E2-only (P5)      - 右下: 联合 native (三专家)
每个面板: 原图 + GT(空心圆) + 该 run 的预测点(实心点，按置信度阈值过滤)。
"""
from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.data.point_dataset import PointDataset
from test_each_dataset import load_checkpoint_model

EXPERT_COLORS = {
    0: "#E63232",  # E0/P3 红
    1: "#28C83C",  # E1/P4 绿
    2: "#3278F0",  # E2/P5 蓝
}


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


def load_models(
    checkpoint_paths: list[str],
    weights: str,
    device: str,
) -> list[tuple[str, object, str, int | None]]:
    entries = []
    for path in checkpoint_paths:
        model, metadata = load_checkpoint_model(path, weights, device)
        config = metadata["config"]
        assert isinstance(config, dict)
        expert_index = config.get("expert_index")
        if expert_index is not None:
            expert_index = int(expert_index)
        routing = "expert_only" if expert_index is not None else "native"
        tag = (
            f"E{expert_index}-only"
            if expert_index is not None
            else "joint"
        )
        model.eval()
        entries.append((tag, model, routing, expert_index))
    return entries


def render_panel(
    ax,
    image: np.ndarray,
    gt: np.ndarray,
    predictions: dict[str, torch.Tensor],
    routing: str,
    conf_threshold: float,
) -> None:
    ax.imshow(image)
    ax.axis("off")
    if gt.shape[0]:
        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            facecolors="none",
            edgecolors="#FFD700",
            linewidths=0.8,
            s=26,
            marker="o",
        )
    logits = predictions["logits"][0].sigmoid().cpu().numpy()
    points = predictions["points"][0].cpu().numpy()
    indices = predictions["expert_indices"][0].cpu().numpy()
    keep = logits > conf_threshold
    if keep.any():
        points = points[keep]
        indices = indices[keep]
        if routing == "expert_only":
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=4,
                c="#FFFFFF",
                edgecolors="none",
            )
        else:
            for expert in range(3):
                mask = indices == expert
                if mask.any():
                    ax.scatter(
                        points[mask, 0],
                        points[mask, 1],
                        s=4,
                        c=EXPERT_COLORS[expert],
                        edgecolors="none",
                    )


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "compare.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("使用设备: %s", device)

    dataset = PointDataset(
        args.data_root,
        split=args.split,
        crop_size=args.imgsz,
        augment=False,
    )
    entries = load_models(args.checkpoint, args.weights, device)
    logging.info(
        "面板顺序: %s",
        " | ".join(tag for tag, _, _, _ in entries),
    )

    selected = (
        list(range(len(dataset)))
        if args.max_images < 0
        else list(range(min(args.max_images, len(dataset))))
    )

    for index in tqdm(selected, desc="生成对比图"):
        sample = dataset[index]
        image = sample["img"]
        gt = sample["points"]
        filename = os.path.basename(dataset.image_paths[index])
        image_np = (
            image.permute(1, 2, 0).numpy()
            if isinstance(image, torch.Tensor)
            else np.asarray(image)
        )
        image_np = np.clip(
            np.asarray(image_np, dtype=np.float32) * 255.0
            if image_np.max() <= 1.5
            else image_np,
            0,
            255,
        ).astype(np.uint8)

        cols = min(len(entries), 2)
        rows = (len(entries) + 1) // 2
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(9 * cols, 9 * rows),
            dpi=150,
        )
        axes = np.asarray(axes).reshape(-1)

        with torch.inference_mode():
            input_image = image.unsqueeze(0).to(device)
            for panel_index, (tag, model, routing, expert_index) in enumerate(
                entries
            ):
                predictions = model(
                    input_image,
                    routing_mode=routing,
                    expert_index=expert_index,
                )
                ax = axes[panel_index]
                render_panel(
                    ax,
                    image_np,
                    gt.numpy() if not isinstance(gt, torch.Tensor) else gt.numpy(),
                    predictions,
                    routing,
                    args.conf_threshold,
                )
                pred_count = float(
                    predictions["logits"].sigmoid().sum()
                )
                ax.set_title(
                    f"{tag} | GT={gt.shape[0]} Pred={pred_count:.1f}",
                    fontsize=13,
                    fontweight="bold",
                    pad=6,
                )

        for extra in range(len(entries), axes.size):
            axes[extra].axis("off")

        fig.suptitle(
            f"{filename} (crop={args.imgsz}, conf>{args.conf_threshold})",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = os.path.join(
            args.out_dir, f"{index:03d}_{os.path.splitext(filename)[0]}.jpg"
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    logging.info("对比图输出目录: %s", args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="同一张 val 图的 E0/E1/E2/联合 横向对比面板"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="依次作为面板 (建议 E0-only/E1-only/E2-only/joint)",
    )
    parser.add_argument("--data-root", type=str, default="datasets/shanghaitech_AB")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=str, default="runs/ablation_eval/compare")
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="生成前 N 张；-1 生成全部",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    main(parse_args())
