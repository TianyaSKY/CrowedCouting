"""重叠滑窗裁切推理 + 余弦窗加权融合。

推理侧不做任何缩放：原图分辨率上按 crop_size 重叠滑窗逐 tile 前向，
各层候选置信度网格用余弦窗加权融合成全图网格。

计数与热力图是两条独立链路（reference 维度不混用）：

    sigmoid logits
        ├─ sum over references → 各层软计数网格（fused_ref_probs）
        │      soft_count = Σ_level fused_ref_probs.sum()
        └─ max over references → 概率热力图网格（fused_levels）

tile 输出按 expert_indices 建 mask，再用该 expert 自己的 stride/refs
reshape 回 [H, W, K]，因此 native 与 expert_only(E0/E1/E2) 走同一套
解码逻辑，不会把 E2 的 20x20x16 错当 80x80x1。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


def cosine_window_1d(size: int) -> np.ndarray:
    """长度 size 的一维余弦坡窗：两端趋近 0，中心为 1，归一化到峰值 1。"""
    if size <= 0:
        raise ValueError("size must be positive")
    positions = np.arange(size, dtype=np.float64)
    window = 0.5 * (1.0 - np.cos(np.pi * (positions + 0.5) / size))
    return (window / window.max()).astype(np.float32)


@dataclass
class TiledResult:
    count: float            # 融合后软计数 = Σ_level fused_ref_probs.sum()
    level_counts: dict[int, float]  # {stride: 该层软计数}
    points: np.ndarray      # [N,2] 原图像素坐标（conf 过滤 + NMS 去重）
    sources: np.ndarray     # [N] expert_indices
    scores: np.ndarray      # [N] sigmoid 置信度
    prob_map: np.ndarray    # [H,W] float32 全分辨率概率热力图
    raw_points: np.ndarray  # [M,2] 阈值/NMS 之前的全部候选（原图坐标）
    raw_scores: np.ndarray  # [M] 原始 sigmoid 置信度
    raw_sources: np.ndarray # [M] expert_indices


def _tile_starts(length: int, size: int, stride: int) -> tuple[list[int], int]:
    """返回该维度的切块起点与 pad 后的长度（只补右/下）。"""
    if length < size:
        return [0], size
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts, length


def _pool_window(window_2d: np.ndarray, level_stride: int) -> np.ndarray:
    """(S,S) 余弦窗经 reshape-mean 池化到 (S//s_l, S//s_l)。"""
    cells = window_2d.shape[0] // level_stride
    trimmed = window_2d[: cells * level_stride, : cells * level_stride]
    return (
        trimmed.reshape(
            cells,
            level_stride,
            cells,
            level_stride,
        )
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def _greedy_nms(
    points: np.ndarray,
    scores: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跨 tile 按分数降序的贪心 NMS（欧氏距离半径），返回保持分数降序。"""
    order = np.argsort(-scores)
    keep: list[int] = []
    for index in order:
        if not keep:
            keep.append(int(index))
            continue
        diffs = points[keep] - points[int(index)]
        if float(np.sqrt((diffs * diffs).sum(axis=1)).min()) >= radius:
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    return points[keep_array], scores[keep_array], keep_array


def expert_grid_shapes(
    model,
    tile_height: int,
    tile_width: int,
) -> dict[int, tuple[int, int, int]]:
    """返回 {expert_index: (cells_h, cells_w, refs)}。

    依据模型 point_head 的 output_strides / references_per_expert 计算，
    tile 内每个 expert 的特征网格大小。
    """
    shapes: dict[int, tuple[int, int, int]] = {}
    for expert_index, (stride, refs) in enumerate(
        zip(
            model.point_head.output_strides,
            model.point_head.references_per_expert,
        )
    ):
        cells_h = tile_height // stride
        cells_w = tile_width // stride
        shapes[int(expert_index)] = (
            int(cells_h),
            int(cells_w),
            int(refs),
        )
    return shapes


def tiled_forward(
    model,
    image_bgr: np.ndarray,
    device: str,
    crop_size: int,
    overlap: float = 0.5,
    tile_batch_size: int = 8,
    routing_mode: str = "native",
    expert_index: int | None = None,
) -> dict:
    """滑窗前向并融合各层 reference 置信度网格。

    返回 {"fused_levels": {stride: [H,W] 热力图(逐 cell max over refs)},
           "fused_ref_probs": {stride: [H,W,K] 软计数网格},
           "soft_count": float, "level_counts": {stride: float},
           "tile_predictions": [...], "origins": [(y0,x0),...],
           "padded_hw": (Hp,Wp)}。
    供训练验证复用（criterion 需要逐 tile 原始输出），不做阈值/NMS。
    """
    height, width = image_bgr.shape[:2]
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    stride = max(1, round(crop_size * (1.0 - overlap)))
    ys, padded_h = _tile_starts(height, crop_size, stride)
    xs, padded_w = _tile_starts(width, crop_size, stride)
    if (padded_h, padded_w) != (height, width):
        image_bgr = cv2.copyMakeBorder(
            image_bgr,
            0,
            padded_h - height,
            0,
            padded_w - width,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

    output_strides = [
        int(value) for value in model.point_head.output_strides
    ]
    references_per_expert = [
        int(value) for value in model.point_head.references_per_expert
    ]

    # 每个 tile 内各 expert 的特征网格尺寸。
    grid_shapes = expert_grid_shapes(model, crop_size, crop_size)
    # 单 tile 的候选总数，用于校验 expert mask 解码结果。
    expected_candidates = sum(
        cells_h * cells_w * refs
        for cells_h, cells_w, refs in grid_shapes.values()
    )
    if routing_mode == "expert_only":
        if expert_index is None or expert_index not in grid_shapes:
            raise ValueError(
                f"routing_mode='expert_only' 时 expert_index 必须是 "
                f"0..{len(grid_shapes) - 1}"
            )
        cells_h, cells_w, refs = grid_shapes[expert_index]
        expected_candidates = cells_h * cells_w * refs

    window_2d = np.outer(
        cosine_window_1d(crop_size),
        cosine_window_1d(crop_size),
    ).astype(np.float32)
    pooled_windows = {
        s_l: _pool_window(window_2d, s_l) for s_l in output_strides
    }

    fused_ref_acc = {
        s_l: np.zeros(
            (
                padded_h // s_l,
                padded_w // s_l,
                references_per_expert[expert],
            ),
            dtype=np.float32,
        )
        for expert, s_l in enumerate(output_strides)
    }
    weight_acc = {
        s_l: np.zeros(
            (padded_h // s_l, padded_w // s_l), dtype=np.float32
        )
        for s_l in output_strides
    }

    tile_tensors: list[torch.Tensor] = []
    origins: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            tile = image_bgr[y0:y0 + crop_size, x0:x0 + crop_size]
            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_tensors.append(
                torch.from_numpy(rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
            )
            origins.append((y0, x0))

    tile_predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(tile_tensors), tile_batch_size):
            batch = torch.stack(
                tile_tensors[start:start + tile_batch_size]
            ).to(device)
            predictions = model(
                batch,
                routing_mode=routing_mode,
                expert_index=expert_index,
            )
            tile_predictions.append(predictions)
            probs = predictions["logits"].sigmoid()
            tile_expert_indices = predictions["expert_indices"]

            for tile_index in range(probs.shape[0]):
                y0, x0 = origins[start + tile_index]
                tile_probs = probs[tile_index].cpu().numpy()
                tile_sources = (
                    tile_expert_indices[tile_index].cpu().numpy()
                )
                if tile_probs.shape[0] != expected_candidates:
                    raise ValueError(
                        f"tile 候选数 {tile_probs.shape[0]} 与配置 "
                        f"{expected_candidates} 不一致；请检查 "
                        f"routing_mode={routing_mode!r} "
                        f"expert_index={expert_index!r}"
                    )

                for expert, s_l in enumerate(output_strides):
                    cells_h, cells_w, refs = grid_shapes[expert]
                    mask = tile_sources == expert
                    if not bool(mask.any()):
                        continue
                    level_scores = tile_probs[mask]
                    if level_scores.size != cells_h * cells_w * refs:
                        raise ValueError(
                            f"E{expert} mask 候选数 {level_scores.size} "
                            f"与网格 {cells_h}x{cells_w}x{refs} 不一致"
                        )
                    conf_grid = level_scores.reshape(
                        cells_h, cells_w, refs
                    ).astype(np.float32)
                    window = pooled_windows[s_l]
                    row = y0 // s_l
                    col = x0 // s_l
                    slice_y = slice(row, row + cells_h)
                    slice_x = slice(col, col + cells_w)
                    fused_ref_acc[s_l][slice_y, slice_x] += (
                        window[:, :, None] * conf_grid
                    )
                    weight_acc[s_l][slice_y, slice_x] += window

    fused_ref_probs = {
        s_l: grid / np.maximum(weight_acc[s_l][:, :, None], 1e-8)
        for s_l, grid in fused_ref_acc.items()
    }
    # 热力图：每个位置取所有 reference 的最高人头概率。
    fused_levels = {
        s_l: grid.max(axis=-1) for s_l, grid in fused_ref_probs.items()
    }
    level_counts = {
        s_l: float(grid.sum()) for s_l, grid in fused_ref_probs.items()
    }
    soft_count = float(sum(level_counts.values()))
    return {
        "fused_levels": fused_levels,
        "fused_ref_probs": fused_ref_probs,
        "soft_count": soft_count,
        "level_counts": level_counts,
        "tile_predictions": tile_predictions,
        "origins": origins,
        "padded_hw": (padded_h, padded_w),
    }


def run_tiled_inference(
    model,
    image_bgr: np.ndarray,
    device: str,
    crop_size: int,
    overlap: float = 0.5,
    tile_batch_size: int = 8,
    conf_threshold: float = 0.5,
    nms_radius: int | None = None,
    routing_mode: str = "native",
    expert_index: int | None = None,
    return_raw_candidates: bool = False,
) -> TiledResult:
    """滑窗推理完整出口：软计数 + 全分辨率概率热力图 + NMS 后点位。

    return_raw_candidates=True 时同时返回阈值/NMS 之前的全部候选
    （raw_points/raw_scores/raw_sources，原图坐标），用于不受阈值
    影响的置信度诊断（如 GT 邻域最大原始置信度）。
    """
    height, width = image_bgr.shape[:2]
    forward = tiled_forward(
        model,
        image_bgr,
        device,
        crop_size,
        overlap=overlap,
        tile_batch_size=tile_batch_size,
        routing_mode=routing_mode,
        expert_index=expert_index,
    )
    fused_levels: dict[int, np.ndarray] = forward["fused_levels"]
    origins: list[tuple[int, int]] = forward["origins"]
    padded_h, padded_w = forward["padded_hw"]

    count = float(forward["soft_count"])
    level_counts = dict(forward["level_counts"])

    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    for grid in fused_levels.values():
        upsampled = cv2.resize(
            grid,
            (padded_w, padded_h),
            interpolation=cv2.INTER_LINEAR,
        )
        prob_map = np.maximum(prob_map, upsampled)
    prob_map = prob_map[:height, :width]

    if nms_radius is None:
        nms_radius = crop_size // 4

    candidate_points: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    candidate_sources: list[np.ndarray] = []
    raw_points_list: list[np.ndarray] = []
    raw_scores_list: list[np.ndarray] = []
    raw_sources_list: list[np.ndarray] = []
    origin_index = 0
    for predictions in forward["tile_predictions"]:
        probs = predictions["logits"].sigmoid()
        points = predictions["points"]
        sources = predictions["expert_indices"]
        for tile_index in range(probs.shape[0]):
            y0, x0 = origins[origin_index]
            origin_index += 1
            if return_raw_candidates:
                raw_points = (
                    points[tile_index].cpu().numpy()
                    + np.asarray([x0, y0], dtype=np.float32)
                )
                raw_points[:, 0] = np.clip(
                    raw_points[:, 0], 0, max(width - 1, 0)
                )
                raw_points[:, 1] = np.clip(
                    raw_points[:, 1], 0, max(height - 1, 0)
                )
                raw_points_list.append(raw_points.astype(np.float32))
                raw_scores_list.append(
                    probs[tile_index].cpu().numpy().astype(np.float32)
                )
                raw_sources_list.append(
                    sources[tile_index].cpu().numpy().astype(np.int64)
                )
            keep = probs[tile_index] > conf_threshold
    origin_index = 0
    for predictions in forward["tile_predictions"]:
        probs = predictions["logits"].sigmoid()
        points = predictions["points"]
        sources = predictions["expert_indices"]
        for tile_index in range(probs.shape[0]):
            y0, x0 = origins[origin_index]
            origin_index += 1
            keep = probs[tile_index] > conf_threshold
            if not bool(keep.any()):
                continue
            selected_points = (
                points[tile_index][keep].cpu().numpy()
                + np.asarray([x0, y0], dtype=np.float32)
            )
            selected_points[:, 0] = np.clip(
                selected_points[:, 0], 0, max(width - 1, 0)
            )
            selected_points[:, 1] = np.clip(
                selected_points[:, 1], 0, max(height - 1, 0)
            )
            candidate_points.append(selected_points.astype(np.float32))
            candidate_scores.append(
                probs[tile_index][keep].cpu().numpy().astype(np.float32)
            )
            candidate_sources.append(
                sources[tile_index][keep].cpu().numpy().astype(np.int64)
            )

    if not candidate_points:
        return TiledResult(
            count=count,
            level_counts=level_counts,
            points=np.zeros((0, 2), dtype=np.float32),
            sources=np.zeros((0,), dtype=np.int64),
            scores=np.zeros((0,), dtype=np.float32),
            prob_map=prob_map,
            raw_points=(
                np.concatenate(raw_points_list, axis=0)
                if raw_points_list
                else np.zeros((0, 2), dtype=np.float32)
            ),
            raw_scores=(
                np.concatenate(raw_scores_list, axis=0)
                if raw_scores_list
                else np.zeros((0,), dtype=np.float32)
            ),
            raw_sources=(
                np.concatenate(raw_sources_list, axis=0)
                if raw_sources_list
                else np.zeros((0,), dtype=np.int64)
            ),
        )

    all_points = np.concatenate(candidate_points, axis=0)
    all_scores = np.concatenate(candidate_scores, axis=0)
    all_sources = np.concatenate(candidate_sources, axis=0)
    kept_points, kept_scores, kept_order = _greedy_nms(
        all_points, all_scores, float(nms_radius)
    )
    return TiledResult(
        count=count,
        level_counts=level_counts,
        points=kept_points.astype(np.float32),
        sources=all_sources[kept_order],
        scores=kept_scores,
        prob_map=prob_map,
        raw_points=(
            np.concatenate(raw_points_list, axis=0)
            if raw_points_list
            else np.zeros((0, 2), dtype=np.float32)
        ),
        raw_scores=(
            np.concatenate(raw_scores_list, axis=0)
            if raw_scores_list
            else np.zeros((0,), dtype=np.float32)
        ),
        raw_sources=(
            np.concatenate(raw_sources_list, axis=0)
            if raw_sources_list
            else np.zeros((0,), dtype=np.int64)
        ),
    )
