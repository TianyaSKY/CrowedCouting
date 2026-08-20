import argparse
import logging
import os

import cv2
import numpy as np
import torch

from scripts.data.point_dataset import (
    inverse_letterbox_points,
    letterbox_image,
)
from models.yolo11_moe_point import YOLO11MoEPoint

# 三个专家的可视化颜色（BGR）：route 0=P3 专家、1=P4 专家、2=P5 专家
EXPERT_COLORS = [
    (0, 0, 255),    # P3 局部细节 - 红色
    (0, 255, 0),    # P4 中层上下文 - 绿色
    (255, 0, 0),    # P5 大范围上下文 - 蓝色
]
def _read_checkpoint(checkpoint_path: str | None):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def resolve_inference_settings(
    checkpoint_path: str | None,
    imgsz: int | None = None,
    temperature: float | None = None,
) -> tuple[int, float]:
    """Resolve image size and Router temperature from checkpoint config."""
    checkpoint = _read_checkpoint(checkpoint_path)
    config = (
        checkpoint.get("config", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    checkpoint_args = (
        checkpoint.get("args", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    if not isinstance(config, dict):
        config = {}
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    resolved_imgsz = (
        imgsz
        if imgsz is not None
        else config.get(
            "crop_size",
            checkpoint_args.get("crop_size", 640),
        )
    )
    resolved_temperature = (
        temperature
        if temperature is not None
        else config.get(
            "temperature",
            checkpoint.get("temperature", 1.0)
            if isinstance(checkpoint, dict)
            else 1.0,
        )
    )
    resolved_imgsz = int(resolved_imgsz)
    resolved_temperature = float(resolved_temperature)
    if resolved_imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if resolved_temperature <= 0:
        raise ValueError("temperature must be positive")
    return resolved_imgsz, resolved_temperature


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _checkpoint_model_settings(
    checkpoint,
    weights_path: str,
) -> tuple[str, int, int]:
    weights = weights_path
    hidden_channels = 256
    num_references = 4
    checkpoint_args = (
        checkpoint.get("args", {})
        if isinstance(checkpoint, dict)
        else {}
    )
    if isinstance(checkpoint_args, dict):
        weights = str(
            checkpoint_args.get("weights", weights_path)
        )
        hidden_channels = int(
            checkpoint_args.get("hidden_channels", 256)
        )
        num_references = int(
            checkpoint_args.get("num_references", 4)
        )
    return weights, hidden_channels, num_references




def load_model(weights_path, checkpoint_path, device):
    """构建模型并加载训练好的权重。"""
    checkpoint = _read_checkpoint(checkpoint_path)
    weights, hidden_channels, num_references = (
        _checkpoint_model_settings(checkpoint, weights_path)
    )

    model = YOLO11MoEPoint(
        weights=weights,
        hidden_channels=hidden_channels,
        num_references=num_references,
    ).to(device)

    if checkpoint is not None:
        logging.info(
            f"从 {checkpoint_path} 加载权重 "
            f"(weights={weights}, hidden_channels={hidden_channels}, "
            f"num_references={num_references})"
        )
        state_dict = _checkpoint_state_dict(checkpoint)
        logging.info(
            f"checkpoint: epoch={checkpoint.get('epoch')} "
            f"best_mae={checkpoint.get('best_mae')}"
        )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            logging.warning(
                f"旧版 checkpoint 缺少新参数({error})，"
                "缺失部分使用初始化值"
            )
            model.load_state_dict(
                state_dict, strict=False
            )
    else:
        logging.warning(
            "未提供权重文件，模型使用随机初始化参数进行推理。"
        )

    model.eval()
    return model


def predict_image(
    model,
    image_bgr: np.ndarray,
    device: str,
    imgsz: int = 640,
    conf_threshold: float = 0.5,
    temperature: float = 1.0,
):
    """对单张图像推理并返回可见点与正式 soft count。

    返回 ``(points, routes, scores, metric_count)``。可见点仍由
    ``conf_threshold`` 筛选，但 ``metric_count`` 始终是所有候选点的
    ``sum(sigmoid(logits))``，与训练期验证一致。
    """
    original_height, original_width = image_bgr.shape[:2]
    padded_bgr, scale, pad_x, pad_y = letterbox_image(
        image_bgr,
        imgsz,
    )
    image_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(
            image_rgb.astype(np.float32) / 255.0
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        predictions = model(
            tensor,
            temperature=temperature,
            routing_mode="top2",
        )

    scores = predictions["logits"].sigmoid()[0]
    points = predictions["points"][0]
    routes = predictions["gates"].argmax(dim=-1)[0]
    metric_count = float(scores.sum().item())

    keep = scores > conf_threshold
    selected_points = points[keep].cpu().numpy()
    selected_routes = routes[keep].cpu().numpy()
    selected_scores = scores[keep].cpu().numpy()

    selected_points = inverse_letterbox_points(
        selected_points,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height,
    )
    return (
        selected_points,
        selected_routes,
        selected_scores,
        metric_count,
    )


def draw_predictions(
    image_bgr: np.ndarray,
    points: np.ndarray,
    routes: np.ndarray,
    metric_count: float | None = None,
) -> np.ndarray:
    """绘制可见预测点并同时显示 soft count 与可见点数。"""
    result = image_bgr.copy()

    for point, route in zip(points, routes):
        x, y = int(round(float(point[0]))), int(
            round(float(point[1]))
        )
        color = EXPERT_COLORS[int(route) % len(EXPERT_COLORS)]
        cv2.circle(result, (x, y), radius=3, color=color, thickness=-1)

    if metric_count is None:
        metric_count = float(len(points))

    # 在图像上方增加独立标题横条，不遮挡图像内容
    header_height = 42
    header = np.full((header_height, result.shape[1], 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(
        header,
        f"Soft Count: {metric_count:.1f} | Visible Points: {len(points)}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, result])


def predict_main(args):
    # 单图脚本独立运行：默认控制台输出（batch 脚本会以 force 覆盖为文件）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"使用设备: {device}")

    model = load_model(args.weights, args.checkpoint, device)
    imgsz, temperature = resolve_inference_settings(
        args.checkpoint,
        imgsz=args.imgsz,
        temperature=args.temperature,
    )
    logging.info(
        "推理设置: imgsz=%d temperature=%.4f",
        imgsz,
        temperature,
    )

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法读取图片 {args.image}")

    points, routes, scores, metric_count = predict_image(
        model,
        image,
        device,
        imgsz=imgsz,
        conf_threshold=args.conf,
        temperature=temperature,
    )

    logging.info(
        "Metric count=%.3f，Visible points=%d",
        metric_count,
        len(points),
    )
    for route in range(3):
        logging.info(
            f"  专家 {route}: {int((routes == route).sum())} 个可见点"
        )

    result = draw_predictions(
        image,
        points,
        routes,
        metric_count=metric_count,
    )
    out_path = args.output or args.image.replace(
        ".jpg", "_moe_pred.jpg"
    )
    cv2.imwrite(out_path, result)
    logging.info(f"已将预测结果保存至 {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 + 点级 Scale-MoE Head 推理与可视化"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="输入图片路径"
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11m.pt",
        help="YOLO11 预训练权重（用于构建 Backbone+Neck）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="runs/moe_point/best_top2.pt",
        help="训练好的 D2 best_top2.pt checkpoint"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="推理输入尺寸（默认读取 checkpoint 的 crop_size）"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Router 推理温度（默认读取 checkpoint 的 config/temperature；"
        "可通过本参数覆盖）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值（仅用于可见点筛选；Metric count 使用 sigmoid 求和）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径"
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_main(parse_args())
