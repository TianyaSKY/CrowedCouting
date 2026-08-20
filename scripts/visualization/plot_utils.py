"""人群计数与点级 MoE 评估可视化与报告生成工具模块。

提供：
1. GT vs Pred 回归散点图 (plot_count_scatter)
2. 跨数据集 MAE / RMSE 对比柱状图 (plot_benchmark_bar_chart)
3. 点级 MoE 定性对比多面板图 (create_moe_comparison_figure)
4. Markdown 自动化基准评测报告生成 (generate_markdown_report)
5. 图像保存安全封装 (save_figure)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # 确保在无 GUI 环境安全绘图
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# 专家分色（RGB 格式）：E0=P3 红色、E1=P4 绿色、E2=P5 蓝色
EXPERT_RGB_COLORS = {
    0: (230, 50, 50),     # P3 红色
    1: (40, 200, 60),     # P4 绿色
    2: (50, 120, 240),    # P5 蓝色
}
GT_RGB_COLOR = (255, 215, 0)  # 金黄色


def tensor_to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """将张量或数组转换为 uint8 RGB NumPy 数组 (H, W, 3)。"""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img_np = img.numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    if np.issubdtype(img_np.dtype, np.floating):
        if img_np.max() <= 1.05 and img_np.min() >= -0.05:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """保存 Matplotlib 图像至指定文件路径并释放资源。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    if close:
        plt.close(fig)


def plot_count_scatter(
    gt_counts: Sequence[float] | torch.Tensor | np.ndarray,
    pred_counts: Sequence[float] | torch.Tensor | np.ndarray,
    metrics: dict[str, float | int] | None = None,
    title: str = "Ground Truth vs. Predicted Count",
    figsize: tuple[float, float] = (6.5, 6.0),
    dpi: int = 150,
) -> plt.Figure:
    """绘制真实人数 vs 预测人数的回归散点图与理想对角线。

    Args:
        gt_counts: 真实人数列表
        pred_counts: 预测人数列表
        metrics: 包含 'mae', 'rmse', 'nae' 等指标的字典（可选）
        title: 图表标题
        figsize: 画布尺寸
        dpi: 分辨率
    """
    if isinstance(gt_counts, torch.Tensor):
        gts = gt_counts.detach().cpu().numpy().reshape(-1)
    else:
        gts = np.asarray(gt_counts, dtype=np.float32).reshape(-1)

    if isinstance(pred_counts, torch.Tensor):
        preds = pred_counts.detach().cpu().numpy().reshape(-1)
    else:
        preds = np.asarray(pred_counts, dtype=np.float32).reshape(-1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(
        gts,
        preds,
        alpha=0.65,
        edgecolors="none",
        c="#1f77b4",
        s=32,
        label="Image Samples",
    )

    min_val = min(float(np.min(gts)) if len(gts) > 0 else 0.0, float(np.min(preds)) if len(preds) > 0 else 0.0, 0.0)
    max_val = max(float(np.max(gts)) if len(gts) > 0 else 1.0, float(np.max(preds)) if len(preds) > 0 else 1.0, 1.0)
    margin = (max_val - min_val) * 0.05
    line_min = max(0.0, min_val - margin)
    line_max = max_val + margin

    # 绘制理想参考线 y = x
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        "r--",
        linewidth=1.8,
        label="Ideal (y = x)",
    )

    ax.set_xlabel("Ground Truth Count", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Count", fontsize=11, fontweight="bold")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 指标说明文本框
    if metrics:
        metric_lines = []
        if "num_images" in metrics or "n_samples" in metrics:
            n = metrics.get("num_images", metrics.get("n_samples"))
            metric_lines.append(f"N: {n}")
        if "mae" in metrics:
            metric_lines.append(f"MAE: {metrics['mae']:.2f}")
        if "rmse" in metrics:
            metric_lines.append(f"RMSE: {metrics['rmse']:.2f}")
        if "nae" in metrics:
            metric_lines.append(f"NAE: {metrics['nae']:.4f}")
        if "gt_total" in metrics and "pred_total" in metrics:
            metric_lines.append(f"GT Total: {int(metrics['gt_total'])}")
            metric_lines.append(f"Pred Total: {float(metrics['pred_total']):.1f}")

        metrics_text = "\n".join(metric_lines)
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.88,
                edgecolor="#cccccc",
            ),
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.88)
    plt.tight_layout()
    return fig


def plot_benchmark_bar_chart(
    dataset_metrics: dict[str, dict[str, Any]],
    save_path: str | Path,
    title: str = "Crowd Counting Cross-Dataset Benchmark Performance",
    dpi: int = 200,
) -> None:
    """绘制跨数据集 MAE / RMSE 对比柱状图。

    Args:
        dataset_metrics: {name: {"mae": float, "rmse": float, ...}}
        save_path: 目标保存路径
        title: 图表标题
        dpi: 分辨率
    """
    valid_names = [name for name, m in dataset_metrics.items() if "mae" in m and "rmse" in m]
    if not valid_names:
        return

    maes = [float(dataset_metrics[name]["mae"]) for name in valid_names]
    rmses = [float(dataset_metrics[name]["rmse"]) for name in valid_names]

    # 添加宏平均柱
    if len(valid_names) > 1 and "AVERAGE" not in valid_names and "★ AVERAGE" not in valid_names:
        display_names = list(valid_names) + ["★ AVERAGE"]
        maes.append(float(np.mean(maes)))
        rmses.append(float(np.mean(rmses)))
    else:
        display_names = list(valid_names)

    x = np.arange(len(display_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(display_names) * 1.8), 5.2), dpi=dpi)
    rects1 = ax.bar(
        x - width / 2,
        maes,
        width,
        label="MAE",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
    )
    rects2 = ax.bar(
        x + width / 2,
        rmses,
        width,
        label="RMSE",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.88,
    )

    ax.set_ylabel("Error (Counts)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha="right", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        if h is not None and not np.isnan(h):
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    for rect in rects2:
        h = rect.get_height()
        if h is not None and not np.isnan(h):
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    plt.tight_layout()
    save_figure(fig, save_path, dpi=dpi)


def draw_points_on_pil(
    image: Image.Image,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 3,
    outline: tuple[int, int, int] | None = (0, 0, 0),
    filled: bool = True,
    width: int = 2,
) -> Image.Image:
    """在 PIL 图像上绘制散点。"""
    draw = ImageDraw.Draw(image)
    r = max(1, int(radius))
    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        box = [(x - r, y - r), (x + r, y + r)]
        if filled:
            draw.ellipse(box, fill=color, outline=outline)
        else:
            draw.ellipse(box, fill=None, outline=color, width=max(1, int(width)))
    return image


def create_moe_comparison_figure(
    image: torch.Tensor | np.ndarray | Image.Image,
    gt_points: np.ndarray | torch.Tensor,
    pred_points: np.ndarray | torch.Tensor,
    pred_routes: np.ndarray | torch.Tensor,
    pred_scores: np.ndarray | torch.Tensor,
    gt_count: float | None = None,
    pred_count: float | None = None,
    title: str | None = None,
    conf_threshold: float = 0.5,
    figsize: tuple[float, float] = (15.5, 5.8),
) -> plt.Figure:
    """生成点级 MoE 定性对比大图（原图+GT / 原图+预测 / GT与预测叠加对齐图 + 图例）。

    Args:
        image: 原图或 letterboxed 图像
        gt_points: 真实点坐标 [N, 2]
        pred_points: 候选预测点坐标 [M, 2]
        pred_routes: 预测点专家分配 [M] (0=P3, 1=P4, 2=P5)
        pred_scores: 预测点置信度分数 [M]
        gt_count: 真实总人数（若为 None 则使用 gt_points 数量）
        pred_count: 预测总人数（若为 None 则使用 soft sum(sigmoid)）
        title: 主标题
        conf_threshold: 可见点置信度阈值
        figsize: 图像尺寸
    """
    from matplotlib.lines import Line2D

    img_np = tensor_to_numpy_image(image)
    pil_gt = Image.fromarray(img_np.copy())
    pil_pred = Image.fromarray(img_np.copy())
    pil_overlay = Image.fromarray(img_np.copy())

    if isinstance(gt_points, torch.Tensor):
        gt_pts = gt_points.detach().cpu().numpy().reshape(-1, 2)
    else:
        gt_pts = np.asarray(gt_points, dtype=np.float32).reshape(-1, 2)

    if isinstance(pred_points, torch.Tensor):
        p_pts = pred_points.detach().cpu().numpy().reshape(-1, 2)
    else:
        p_pts = np.asarray(pred_points, dtype=np.float32).reshape(-1, 2)

    if isinstance(pred_routes, torch.Tensor):
        p_routes = pred_routes.detach().cpu().numpy().reshape(-1)
    else:
        p_routes = np.asarray(pred_routes, dtype=np.int64).reshape(-1)

    if isinstance(pred_scores, torch.Tensor):
        p_scores = pred_scores.detach().cpu().numpy().reshape(-1)
    else:
        p_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)

    if gt_count is None:
        gt_count = float(len(gt_pts))
    if pred_count is None:
        pred_count = float(p_scores.sum()) if len(p_scores) > 0 else 0.0

    rad = max(2, min(img_np.shape[:2]) // 180)

    # 1. 绘制 GT 面板（黄色圆圈）
    draw_points_on_pil(pil_gt, gt_pts, color=GT_RGB_COLOR, radius=rad, outline=(0, 0, 0), filled=True)

    # 2. 绘制预测面板（过滤 conf_threshold，按专家上色）
    keep = p_scores > conf_threshold
    vis_pts = p_pts[keep]
    vis_routes = p_routes[keep]

    route_counts = {0: 0, 1: 0, 2: 0}
    for pt, r in zip(vis_pts, vis_routes):
        r_int = int(r) % 3
        route_counts[r_int] += 1
        color = EXPERT_RGB_COLORS.get(r_int, (255, 0, 0))
        draw_points_on_pil(pil_pred, np.asarray([pt]), color=color, radius=max(2, rad - 1), outline=(0, 0, 0), filled=True)

    # 3. 绘制叠加面板（Overlay：GT 绘制为金色圆环，预测绘制为实心彩色圆点）
    draw_points_on_pil(pil_overlay, gt_pts, color=GT_RGB_COLOR, radius=rad + 2, outline=None, filled=False, width=2)
    for pt, r in zip(vis_pts, vis_routes):
        r_int = int(r) % 3
        color = EXPERT_RGB_COLORS.get(r_int, (255, 0, 0))
        draw_points_on_pil(pil_overlay, np.asarray([pt]), color=color, radius=max(2, rad - 1), outline=(0, 0, 0), filled=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=150)

    # 面板 1: Ground Truth
    axes[0].imshow(np.array(pil_gt))
    axes[0].set_title(f"Ground Truth (Count: {gt_count:.1f})", fontsize=11, fontweight="bold", color="darkgreen", pad=8)
    axes[0].axis("off")

    # 面板 2: MoE Prediction
    err = abs(pred_count - gt_count)
    vis_cnt = len(vis_pts)
    axes[1].imshow(np.array(pil_pred))
    pred_title = f"MoE Prediction (Soft: {pred_count:.1f} | Vis: {vis_cnt})"
    axes[1].set_title(pred_title, fontsize=11, fontweight="bold", color="crimson", pad=8)
    axes[1].axis("off")

    # 面板 3: GT vs. Pred Overlay
    axes[2].imshow(np.array(pil_overlay))
    overlay_title = f"GT vs. Pred Overlay (Err: {err:.1f})"
    axes[2].set_title(overlay_title, fontsize=11, fontweight="bold", color="#2b4c7e", pad=8)
    axes[2].axis("off")

    # 4. 添加规范的 Matplotlib 全局图例 (Legend)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700", markeredgecolor="black", markersize=9, label=f"GT Points ({len(gt_pts)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E63232", markeredgecolor="black", markersize=8, label=f"Expert 0 / P3 ({route_counts[0]})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#28C83C", markeredgecolor="black", markersize=8, label=f"Expert 1 / P4 ({route_counts[1]})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3278F0", markeredgecolor="black", markersize=8, label=f"Expert 2 / P5 ({route_counts[2]})"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9.5,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    return fig


def generate_markdown_report(
    title: str,
    meta: dict[str, Any],
    headers: list[str],
    rows: list[list[str]],
    dataset_details: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """生成格式规范的 Markdown 基准评测总结报告。"""
    md_content = [
        f"# {title}\n",
    ]

    for k, v in meta.items():
        md_content.append(f"- **{k}**: `{v}`")
    md_content.append("")

    md_content.append("## 1. 跨数据集总体指标汇总\n")
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    md_rows = ["| " + " | ".join(r) + " |" for r in rows]
    md_content.extend([md_header, md_sep] + md_rows + ["\n"])

    md_content.append("## 2. 逐数据集详情与产物链接\n")
    for d in dataset_details:
        name = d.get("name", "Unknown")
        md_content.append(f"### 数据集: `{name}`")
        if "description" in d and d["description"]:
            md_content.append(f"- **说明**: {d['description']}")
        if "samples" in d:
            md_content.append(f"- **样本规模**: {d['samples']}")
        if "mae" in d:
            md_content.append(f"- **MAE**: `{d['mae']:.3f}`")
        if "rmse" in d:
            md_content.append(f"- **RMSE**: `{d['rmse']:.3f}`")
        if "nae" in d:
            md_content.append(f"- **NAE**: `{d['nae']:.4f}`")
        if "gt_total" in d and "pred_total" in d:
            md_content.append(f"- **真实总人数**: `{d['gt_total']}` | **预测总人数**: `{d['pred_total']:.1f}`")
        if "artifacts" in d and d["artifacts"]:
            md_content.append("- **产物文件**:")
            for art_name, art_path in d["artifacts"].items():
                md_content.append(f"  - {art_name}: `{art_path}`")
        md_content.append("")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(md_content), encoding="utf-8")
