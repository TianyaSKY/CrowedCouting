import argparse
import logging
import os

import cv2
import numpy as np
import torch

from models.yolo11_moe_point import YOLO11MoEPoint
from scripts.inference.tiling import run_tiled_inference

NATIVE_ARCHITECTURE = "native_multiscale"
EXPERT_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
]


def _read_checkpoint(checkpoint_path: str | None):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def _checkpoint_settings(
    checkpoint,
    weights_path: str,
) -> tuple[str, int, tuple[int, int, int]]:
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Native 推理必须提供 native_multiscale checkpoint"
        )
    config = checkpoint.get("config", {})
    if not isinstance(config, dict) or config.get("architecture") != NATIVE_ARCHITECTURE:
        raise ValueError(
            "只支持 native_multiscale checkpoint；旧 D2 checkpoint 已删除"
        )
    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    weights = str(checkpoint_args.get("weights", weights_path))
    hidden_channels = int(
        config.get("hidden_channels", checkpoint_args.get("hidden_channels", 256))
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
    return weights, hidden_channels, (references[0], references[1], references[2])


def _routing_settings(
    checkpoint,
) -> tuple[str, int | None]:
    """从 checkpoint config 恢复推理路由配置。

    返回 (routing_mode, expert_index)；native checkpoint 返回
    ("native", None)，expert_only checkpoint 返回 ("expert_only", 2) 等。
    """
    if not isinstance(checkpoint, dict):
        return "native", None
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return "native", None
    routing_mode = str(config.get("routing_mode", "native"))
    if routing_mode not in {"native", "expert_only"}:
        raise ValueError(
            f"checkpoint 记录了未知 routing_mode={routing_mode!r}"
        )
    expert_index = config.get("expert_index")
    if expert_index is not None:
        expert_index = int(expert_index)
        if not 0 <= expert_index <= 2:
            raise ValueError(
                f"checkpoint expert_index={expert_index} 超出 0..2"
            )
    if routing_mode == "expert_only" and expert_index is None:
        raise ValueError(
            "checkpoint routing_mode=expert_only 但缺少 expert_index"
        )
    return routing_mode, expert_index


def resolve_inference_settings(
    checkpoint_path: str | None,
    imgsz: int | None = None,
) -> int:
    checkpoint = _read_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(config, dict):
        config = {}
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    resolved_imgsz = imgsz
    if resolved_imgsz is None:
        resolved_imgsz = config.get(
            "crop_size",
            checkpoint_args.get("crop_size", 640),
        )
    resolved_imgsz = int(resolved_imgsz)
    if resolved_imgsz <= 0:
        raise ValueError("imgsz must be positive")
    return resolved_imgsz


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model(weights_path, checkpoint_path, device):
    """加载 checkpoint 并恢复其记录的推理路由配置。

    返回 (model, metadata)，metadata 含 routing_mode / expert_index，
    供调用方显式传给 tiled inference，避免 E2-only checkpoint
    回落到默认 native 三专家推理。
    """
    checkpoint = _read_checkpoint(checkpoint_path)
    weights, hidden_channels, native_references = _checkpoint_settings(
        checkpoint,
        weights_path,
    )
    model = YOLO11MoEPoint(
        weights=weights,
        hidden_channels=hidden_channels,
        native_references=native_references,
    ).to(device)
    state_dict = _checkpoint_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint 中缺少 model state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    routing_mode, expert_index = _routing_settings(checkpoint)
    logging.info(
        "从 %s 加载 native_multiscale 权重 (weights=%s, hidden=%d, refs=%s, "
        "routing_mode=%s, expert_index=%s)",
        checkpoint_path,
        weights,
        hidden_channels,
        native_references,
        routing_mode,
        expert_index,
    )
    return model, {
        "routing_mode": routing_mode,
        "expert_index": expert_index,
    }


def overlay_probability(
    image_bgr: np.ndarray,
    prob_map: np.ndarray,
    alpha: float,
) -> np.ndarray:
    normalized = cv2.normalize(
        prob_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )
    heat = cv2.applyColorMap(
        normalized.astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat, alpha, 0)


def predict_image(
    model,
    image_bgr: np.ndarray,
    device: str,
    imgsz: int = 640,
    conf_threshold: float = 0.5,
    overlap: float = 0.5,
    tile_batch_size: int = 8,
    routing_mode: str = "native",
    expert_index: int | None = None,
):
    result = run_tiled_inference(
        model,
        image_bgr,
        device,
        imgsz,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        conf_threshold=conf_threshold,
        routing_mode=routing_mode,
        expert_index=expert_index,
    )
    return (
        result.points,
        result.sources,
        result.scores,
        result.count,
        result.prob_map,
    )


def draw_predictions(
    image_bgr: np.ndarray,
    points: np.ndarray,
    sources: np.ndarray,
    metric_count: float | None = None,
    routing_mode: str = "native",
    expert_index: int | None = None,
) -> np.ndarray:
    result = image_bgr.copy()
    for point, source in zip(points, sources):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        color = EXPERT_COLORS[int(source) % len(EXPERT_COLORS)]
        cv2.circle(result, (x, y), radius=3, color=color, thickness=-1)
    if metric_count is None:
        metric_count = float(len(points))
    if routing_mode == "expert_only":
        mode_label = f"E{expert_index}-only"
    else:
        mode_label = "Native"
    header = np.full(
        (42, result.shape[1], 3),
        (24, 24, 24),
        dtype=np.uint8,
    )
    cv2.putText(
        header,
        f"{mode_label} Soft Count: {metric_count:.1f} | Visible Points: {len(points)}",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, result])


def predict_main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_model(args.weights, args.checkpoint, device)
    routing_mode = metadata["routing_mode"]
    expert_index = metadata["expert_index"]
    if args.expert_index is not None:
        expert_index = int(args.expert_index)
        routing_mode = (
            "expert_only" if expert_index is not None else "native"
        )
        logging.info(
            "命令行覆盖推理路由: routing_mode=%s expert_index=%s",
            routing_mode,
            expert_index,
        )
    imgsz = resolve_inference_settings(args.checkpoint, imgsz=args.imgsz)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法读取图片 {args.image}")
    points, sources, _, metric_count, prob_map = predict_image(
        model,
        image,
        device,
        imgsz=imgsz,
        conf_threshold=args.conf,
        overlap=args.overlap,
        tile_batch_size=args.tile_batch_size,
        routing_mode=routing_mode,
        expert_index=expert_index,
    )
    logging.info(
        "%s count=%.3f，Visible points=%d",
        f"E{expert_index}-only" if routing_mode == "expert_only" else "Native",
        metric_count,
        len(points),
    )
    for expert_index_loop in range(3):
        logging.info(
            "  专家 %d: %d 个可见点",
            expert_index_loop,
            int((sources == expert_index_loop).sum()),
        )
    result = draw_predictions(
        image,
        points,
        sources,
        metric_count,
        routing_mode=routing_mode,
        expert_index=expert_index,
    )
    stem = os.path.splitext(args.output or args.image)[0]
    out_path = f"{stem}_pred.jpg"
    prob_overlay_path = f"{stem}_prob.jpg"
    prob_raw_path = f"{stem}_prob_raw.png"
    cv2.imwrite(out_path, result)
    cv2.imwrite(
        prob_overlay_path,
        overlay_probability(image, prob_map, args.heat_alpha),
    )
    normalized = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(
        prob_raw_path,
        cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET),
    )
    logging.info("已将预测结果保存至 %s", out_path)
    logging.info("概率叠加图: %s", prob_overlay_path)
    logging.info("独立伪彩概率图: %s", prob_raw_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="native_multiscale 单图推理"
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="checkpoint 未记录 backbone 权重时的 fallback",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/native_multiscale/best_native.pt",
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--heat-alpha", type=float, default=0.45)
    parser.add_argument(
        "--expert-index",
        type=int,
        default=None,
        help="覆盖 checkpoint 记录的 expert_index（0/1/2）；缺省自动恢复",
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_main(parse_args())
