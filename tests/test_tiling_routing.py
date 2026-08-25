"""Correctness tests for tiled inference routing, soft count, and grid decode.

覆盖 Commit A 的回归点：
- expert_only(E2) tiled forward 绝不能出现 E0/E1 候选；
- native 模式仍同时包含 0/1/2；
- 单 tile 软计数必须等于 sum(sigmoid(logits))；
- E2 K=16 的高 logit cell 必须在正确原图位置形成热区峰值
  （防止 20x20x16 被误解码为 80x80x1 回归）；
- overlap 融合后计数不应因同一内容多次出现而翻倍；
- E2 checkpoint 自动恢复 routing_mode=expert_only / expert_index=2。
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference.tiling import run_tiled_inference, tiled_forward
from scripts.visualization.predict_moe import (
    _routing_settings,
    load_model,
)

DEVICE = "cpu"
CROP = 640


class FakePointHead:
    output_strides = (8, 16, 32)
    references_per_expert = (1, 4, 16)


class FakePointModel(torch.nn.Module):
    """模拟 YOLO11MoEPoint 推理契约的合成模型。

    forward(image, routing_mode, expert_index) 返回与真实模型相同的
    logits / points / expert_indices 结构；每个 expert 的候选按
    [h, w, K] 展平，expert_indices 每块为常数。
    """

    architecture = "native_multiscale"

    def __init__(self, base_logit: float = -5.0) -> None:
        super().__init__()
        self.point_head = FakePointHead()
        self.base_logit = base_logit
        # (expert, cell_y, cell_x, ref) -> logit
        self.high: dict[tuple[int, int, int, int], float] = {}
        self.calls: list[tuple[str, int | None]] = []

    def set_cell(
        self,
        expert: int,
        cell_y: int,
        cell_x: int,
        ref: int,
        logit: float,
    ) -> None:
        self.high[(expert, cell_y, cell_x, ref)] = logit

    def forward(
        self,
        image: torch.Tensor,
        routing_mode: str = "native",
        expert_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        self.calls.append((routing_mode, expert_index))
        batch, _, height, width = image.shape
        logits_list: list[torch.Tensor] = []
        points_list: list[torch.Tensor] = []
        indices_list: list[torch.Tensor] = []
        for expert, (stride, refs) in enumerate(
            zip(
                self.point_head.output_strides,
                self.point_head.references_per_expert,
            )
        ):
            if routing_mode == "expert_only" and expert != expert_index:
                continue
            cells_h = height // stride
            cells_w = width // stride
            logits = torch.full(
                (batch, cells_h, cells_w, refs),
                self.base_logit,
                dtype=image.dtype,
            )
            for (ex, cell_y, cell_x, ref), value in self.high.items():
                if (
                    ex == expert
                    and cell_y < cells_h
                    and cell_x < cells_w
                    and ref < refs
                ):
                    logits[:, cell_y, cell_x, ref] = value
            side = int(math.isqrt(refs))
            positions = (
                torch.arange(side, dtype=torch.float32) + 0.5
            ) / side
            ref_y, ref_x = torch.meshgrid(
                positions, positions, indexing="ij"
            )
            reference_offsets = torch.stack(
                [ref_x, ref_y], dim=-1
            ).reshape(refs, 2)
            grid_y = torch.arange(cells_h, dtype=torch.float32)
            grid_x = torch.arange(cells_w, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(
                grid_y, grid_x, indexing="ij"
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)
            base_points = (
                (grid.unsqueeze(2) + reference_offsets) * stride
            ).reshape(-1, 2)
            logits_flat = logits.reshape(batch, -1)
            points = base_points.unsqueeze(0).expand(batch, -1, -1)
            indices = torch.full(
                (batch, logits_flat.shape[1]),
                expert,
                dtype=torch.long,
            )
            logits_list.append(logits_flat)
            points_list.append(points)
            indices_list.append(indices)
        return {
            "architecture": self.architecture,
            "logits": torch.cat(logits_list, dim=1),
            "points": torch.cat(points_list, dim=1),
            "base_points": torch.cat(points_list, dim=1),
            "expert_indices": torch.cat(indices_list, dim=1),
            "source_expert": torch.cat(indices_list, dim=1),
            "references_per_expert": self.point_head.references_per_expert,
            "output_strides": self.point_head.output_strides,
        }


def blank_image(size: int) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def all_tile_indices(forward: dict) -> list[torch.Tensor]:
    return [
        predictions["expert_indices"]
        for predictions in forward["tile_predictions"]
    ]


def test_expert_only_forward_never_contains_other_experts():
    model = FakePointModel()
    model.set_cell(2, 10, 12, 5, 10.0)
    forward = tiled_forward(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode="expert_only",
        expert_index=2,
    )
    assert forward["tile_predictions"], "至少一个 tile 预测"
    for indices in all_tile_indices(forward):
        unique = indices.unique().tolist()
        assert unique == [2], f"E2-only 出现非 E2 候选: {unique}"
    assert all(
        call == ("expert_only", 2) for call in model.calls
    ), "routing 参数没有传递到 model(batch)"
    # 只有 E2 层有候选：其余层 soft count 必须为 0。
    assert forward["level_counts"][8] == 0.0
    assert forward["level_counts"][16] == 0.0
    assert forward["level_counts"][32] > 0.0


def test_native_forward_still_contains_all_experts():
    model = FakePointModel()
    model.set_cell(0, 10, 10, 0, 10.0)
    model.set_cell(1, 20, 20, 1, 10.0)
    model.set_cell(2, 10, 12, 5, 10.0)
    forward = tiled_forward(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode="native",
    )
    unique = torch.cat(all_tile_indices(forward)).unique().tolist()
    assert unique == [0, 1, 2], f"native 模式专家集合异常: {unique}"
    for stride in (8, 16, 32):
        assert forward["level_counts"][stride] > 0.0


def test_single_tile_soft_count_equals_sum_sigmoid_expert_only():
    model = FakePointModel(base_logit=-2.0)
    model.set_cell(2, 5, 5, 3, 4.0)
    forward = tiled_forward(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode="expert_only",
        expert_index=2,
    )
    direct = model(
        torch.zeros(1, 3, CROP, CROP),
        routing_mode="expert_only",
        expert_index=2,
    )
    expected = float(direct["logits"].sigmoid().sum())
    assert math.isclose(
        forward["soft_count"], expected, rel_tol=1e-4
    ), f"soft_count={forward['soft_count']} != sum(sigmoid)={expected}"


def test_single_tile_soft_count_equals_sum_sigmoid_native():
    model = FakePointModel(base_logit=-2.0)
    forward = tiled_forward(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode="native",
    )
    direct = model(
        torch.zeros(1, 3, CROP, CROP),
        routing_mode="native",
    )
    expected = float(direct["logits"].sigmoid().sum())
    assert math.isclose(
        forward["soft_count"], expected, rel_tol=1e-4
    ), f"soft_count={forward['soft_count']} != sum(sigmoid)={expected}"


def test_heatmap_peak_lands_on_p5_reference_position():
    """P5 20x20x16：只把 cell=(10,12), ref=5 的 logit 设高。

    峰值必须落在该 reference 的 P5 原图位置所在 cell 内：
    cell (10,12) 覆盖 y∈[320,352), x∈[384,416)，中心 (336,400)。
    防止 20x20x16 被误解码为 80x80x1 的回归。
    """
    model = FakePointModel(base_logit=-5.0)
    model.set_cell(2, 10, 12, 5, 10.0)
    result = run_tiled_inference(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode="expert_only",
        expert_index=2,
    )
    prob_map = result.prob_map
    assert prob_map.shape == (CROP, CROP)
    flat_index = int(np.argmax(prob_map))
    peak_y, peak_x = divmod(flat_index, CROP)
    assert 320 <= peak_y < 352, f"峰值 y={peak_y} 不在 cell(10,12) 内"
    assert 384 <= peak_x < 416, f"峰值 x={peak_x} 不在 cell(10,12) 内"
    assert abs(peak_y - 336) <= 3 and abs(peak_x - 400) <= 3
    # 峰值置信度必须来自该高 logit（≈sigmoid(10)），而非背景。
    assert prob_map[peak_y, peak_x] > 0.9


def test_overlap_fusion_does_not_double_count():
    """常量 logits 场下，多 tile overlap 融合的软计数 ≈ 单 tile 计数。"""
    model = FakePointModel(base_logit=-2.0)
    image = blank_image(480)
    single = tiled_forward(
        model,
        image,
        DEVICE,
        480,
        overlap=0.5,
        routing_mode="native",
    )
    multi = tiled_forward(
        model,
        image,
        DEVICE,
        320,
        overlap=0.5,
        routing_mode="native",
    )
    assert multi["soft_count"] > 0
    relative = abs(multi["soft_count"] - single["soft_count"]) / single[
        "soft_count"
    ]
    assert relative < 1e-3, (
        f"overlap 融合计数 {multi['soft_count']} 与单 tile "
        f"{single['soft_count']} 偏差 {relative:.4f}（疑似翻倍）"
    )


def test_run_tiled_inference_expert_only_sources_all_e2():
    model = FakePointModel()
    model.set_cell(2, 10, 12, 5, 10.0)
    result = run_tiled_inference(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        conf_threshold=0.5,
        routing_mode="expert_only",
        expert_index=2,
    )
    assert result.points.shape[0] > 0
    assert result.sources.tolist() == [2] * result.points.shape[0]
    assert result.level_counts[32] > 0.0
    assert result.level_counts[8] == 0.0
    assert result.level_counts[16] == 0.0
    assert result.count > 0


def test_routing_settings_native_checkpoint():
    checkpoint = {
        "config": {
            "architecture": "native_multiscale",
            "routing_mode": "native",
        }
    }
    assert _routing_settings(checkpoint) == ("native", None)


def test_routing_settings_expert_only_checkpoint():
    checkpoint = {
        "config": {
            "architecture": "native_multiscale",
            "routing_mode": "expert_only",
            "expert_index": 2,
        }
    }
    assert _routing_settings(checkpoint) == ("expert_only", 2)


def test_routing_settings_missing_expert_index_raises():
    checkpoint = {
        "config": {
            "architecture": "native_multiscale",
            "routing_mode": "expert_only",
        }
    }
    with pytest.raises(ValueError):
        _routing_settings(checkpoint)


def test_raw_candidates_include_below_threshold():
    """P1: RawMaxConf 必须来自阈值/NMS 之前的原始候选。

    低于 conf 阈值的候选不应进入 result.points，但必须出现在
    raw_points/raw_scores 中（否则 0.02/0.08/0.30 的置信度差异
    会被阈值全部抹成 0）。
    """
    model = FakePointModel(base_logit=-8.0)
    model.set_cell(2, 10, 12, 5, -0.6)  # sigmoid(-0.6) ≈ 0.354 < 0.5
    result = run_tiled_inference(
        model,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        conf_threshold=0.5,
        routing_mode="expert_only",
        expert_index=2,
        return_raw_candidates=True,
    )
    assert result.points.shape[0] == 0
    assert result.raw_points.shape[0] == 20 * 20 * 16
    assert result.raw_scores.max() == pytest.approx(
        float(torch.sigmoid(torch.tensor(-0.6))),
        rel=1e-5,
    )
    assert result.raw_sources.tolist() == [2] * result.raw_points.shape[0]


@pytest.mark.skipif(
    not os.path.exists("yolo11n.pt"),
    reason="需要本地 yolo11n.pt",
)
def test_e2_checkpoint_restores_routing_and_tiled_forward(tmp_path):
    """真实模型链路：E2 checkpoint → load_model 恢复 expert_only/2 → tiled 推理。"""
    from models.yolo11_moe_point import YOLO11MoEPoint

    model = YOLO11MoEPoint(
        weights="yolo11n.pt",
        hidden_channels=256,
        native_references=(1, 4, 16),
    )
    checkpoint_path = tmp_path / "best_native.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "architecture": "native_multiscale",
                "routing_mode": "expert_only",
                "expert_index": 2,
                "crop_size": 640,
                "hidden_channels": 256,
                "native_references": [1, 4, 16],
            },
            "args": {"weights": "yolo11n.pt"},
        },
        checkpoint_path,
    )
    loaded, metadata = load_model("yolo11n.pt", str(checkpoint_path), DEVICE)
    assert metadata == {
        "routing_mode": "expert_only",
        "expert_index": 2,
    }
    result = run_tiled_inference(
        loaded,
        blank_image(CROP),
        DEVICE,
        CROP,
        overlap=0.5,
        routing_mode=metadata["routing_mode"],
        expert_index=metadata["expert_index"],
    )
    sources = result.sources
    if sources.size > 0:
        assert sources.tolist() == [2] * sources.size
    # 软计数与直接 expert_only 前向一致。
    with torch.inference_mode():
        direct = loaded(
            torch.zeros(1, 3, CROP, CROP),
            routing_mode="expert_only",
            expert_index=2,
        )
    expected = float(direct["logits"].sigmoid().sum())
    assert math.isclose(result.count, expected, rel_tol=1e-4)
