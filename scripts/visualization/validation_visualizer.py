from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# OpenCV uses BGR: E0=P3 red, E1=P4 green, E2=P5 blue.
EXPERT_COLORS = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)
GT_COLOR = (0, 255, 255)
_TEXT_COLOR = (245, 245, 245)
_TEXT_BACKGROUND = (24, 24, 24)


def _image_to_bgr(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image_array = image.detach().cpu().float().numpy()
    else:
        image_array = np.asarray(image)

    if image_array.ndim == 3 and image_array.shape[0] in (1, 3):
        image_array = np.transpose(image_array, (1, 2, 0))
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            "validation image must have shape [3,H,W] or [H,W,3]"
        )

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0.0, 1.0) * 255.0
        image_array = image_array.astype(np.uint8)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def _points_to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _prediction_tensors(
    predictions: Mapping[str, torch.Tensor | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = predictions["logits"]
    points = predictions["points"]
    gates = predictions["gates"]
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu()
    if isinstance(gates, torch.Tensor):
        gates = gates.detach().cpu()

    logits_array = np.asarray(logits, dtype=np.float32)
    points_array = np.asarray(points, dtype=np.float32)
    gates_array = np.asarray(gates, dtype=np.float32)
    if logits_array.ndim == 2:
        logits_array = logits_array[0]
    if points_array.ndim == 3:
        points_array = points_array[0]
    if gates_array.ndim == 3:
        gates_array = gates_array[0]
    return (
        1.0 / (1.0 + np.exp(-logits_array)),
        points_array.reshape(-1, 2),
        gates_array.reshape(-1, gates_array.shape[-1]),
    )


def _draw_points(
    image_bgr: np.ndarray,
    points: np.ndarray,
    colors: (
        tuple[int, int, int]
        | Sequence[tuple[int, int, int]]
    ),
    *,
    filled: bool,
    radius: int,
) -> None:
    height, width = image_bgr.shape[:2]
    single_color = (
        isinstance(colors, tuple)
        and len(colors) == 3
        and isinstance(colors[0], int)
    )
    for index, point in enumerate(points):
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        color = (
            colors
            if single_color
            else colors[index % len(colors)]
        )
        cv2.circle(
            image_bgr,
            (x, y),
            radius=radius,
            color=color,
            thickness=-1 if filled else 2,
        )


def _add_header_banner(
    image_bgr: np.ndarray,
    lines: Sequence[str],
    min_lines: int = 3,
) -> np.ndarray:
    """Add a header canvas banner above the image instead of drawing on top of image pixels."""
    line_height = 24
    total_lines = max(len(lines), min_lines)
    header_height = 10 + line_height * total_lines
    width = image_bgr.shape[1]

    header = np.full(
        (header_height, width, 3),
        _TEXT_BACKGROUND,
        dtype=np.uint8,
    )
    for line_index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (10, 22 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return np.vstack([header, image_bgr])


def render_validation_sample(
    image: torch.Tensor | np.ndarray,
    gt_points: torch.Tensor | np.ndarray,
    predictions: Mapping[str, torch.Tensor | np.ndarray],
    image_path: str | None = None,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """Render one existing Top-2 validation prediction as an RGB panel.

    The left panel draws all GT points. The right panel only draws points above
    ``conf_threshold`` for readability, while its metric count is the exact
    validation count ``sum(sigmoid(logits))``.
    """
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0 and 1")

    base_bgr = _image_to_bgr(image)
    gt_array = _points_to_numpy(gt_points)
    scores, pred_points, gates = _prediction_tensors(predictions)
    visible = scores > conf_threshold
    visible_points = pred_points[visible]
    visible_routes = gates[visible].argmax(axis=-1)

    gt_panel = base_bgr.copy()
    _draw_points(
        gt_panel,
        gt_array,
        GT_COLOR,
        filled=False,
        radius=max(3, min(gt_panel.shape[:2]) // 160),
    )
    filename = os.path.basename(image_path) if image_path else "validation sample"
    gt_panel = _add_header_banner(
        gt_panel,
        (
            "Ground Truth",
            f"GT count: {len(gt_array)} | {filename}",
        ),
        min_lines=3,
    )

    prediction_panel = base_bgr.copy()
    for point, route in zip(visible_points, visible_routes):
        _draw_points(
            prediction_panel,
            np.asarray([point]),
            (EXPERT_COLORS[int(route) % len(EXPERT_COLORS)],),
            filled=True,
            radius=max(2, min(prediction_panel.shape[:2]) // 220),
        )
    metric_count = float(scores.sum())
    visible_count = int(visible.sum())
    abs_error = abs(metric_count - len(gt_array))
    prediction_panel = _add_header_banner(
        prediction_panel,
        (
            "Prediction - deterministic Top-2",
            f"Metric count: {metric_count:.1f} | Visible points: {visible_count}",
            f"Abs error: {abs_error:.1f} | conf > {conf_threshold:.2f}",
        ),
        min_lines=3,
    )

    panel = np.concatenate((gt_panel, prediction_panel), axis=1)
    return cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)


def render_validation_batch(
    samples: Sequence[Mapping[str, object]],
    conf_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Render a fixed validation sample collection without another forward."""
    return [
        render_validation_sample(
            sample["image"],  # type: ignore[arg-type]
            sample["gt_points"],  # type: ignore[arg-type]
            sample["predictions"],  # type: ignore[arg-type]
            image_path=sample.get("image_path"),  # type: ignore[arg-type]
            conf_threshold=conf_threshold,
        )
        for sample in samples
    ]


def log_validation_images(
    writer: SummaryWriter,
    tag_prefix: str,
    samples: Sequence[Mapping[str, object]],
    epoch: int,
    conf_threshold: float = 0.5,
) -> None:
    """Write rendered samples under ``<tag_prefix>/sample_NN``."""
    for sample_index, image in enumerate(
        render_validation_batch(samples, conf_threshold=conf_threshold)
    ):
        writer.add_image(
            f"{tag_prefix}/sample_{sample_index:02d}",
            image,
            global_step=epoch,
            dataformats="HWC",
        )
