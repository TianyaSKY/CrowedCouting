"""Integration test for train_moe.evaluate_native_count_mae.

覆盖 P0 回归：验证循环必须完整走
    image -> load_points(GT 原图坐标) -> tiled_forward -> soft count
-> criterion(逐 tile GT) -> run_tiled_inference(可视化样本)

此前 `gt_points_np = load_points(...)` 缺失导致首次 validation
直接 NameError，本测试直接调用该函数确保不回归。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.point_moe_loss import PointMoELoss
from scripts.data.point_dataset import PointDataset
from scripts.training.train_moe import evaluate_native_count_mae

CROP = 256


class FakePointHead:
    output_strides = (8, 16, 32)
    references_per_expert = (1, 4, 16)


class FakePointModel(torch.nn.Module):
    """无需外部权重的 native_multiscale 推理契约。"""

    def __init__(self, logit: float = -5.0) -> None:
        super().__init__()
        self.point_head = FakePointHead()
        self.logit = logit

    def forward(
        self,
        image: torch.Tensor,
        routing_mode: str = "native",
        expert_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
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
                self.logit,
                dtype=image.dtype,
                device=image.device,
            )
            side = int(refs**0.5)
            positions = (
                torch.arange(
                    side,
                    dtype=image.dtype,
                    device=image.device,
                )
                + 0.5
            ) / side
            ref_y, ref_x = torch.meshgrid(
                positions,
                positions,
                indexing="ij",
            )
            reference_offsets = torch.stack(
                [ref_x, ref_y],
                dim=-1,
            ).reshape(refs, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(
                    cells_h,
                    dtype=image.dtype,
                    device=image.device,
                ),
                torch.arange(
                    cells_w,
                    dtype=image.dtype,
                    device=image.device,
                ),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)
            base_points = (
                (grid.unsqueeze(2) + reference_offsets) * stride
            ).reshape(-1, 2)
            logits_flat = logits.reshape(batch, -1)
            logits_list.append(logits_flat)
            points_list.append(
                base_points.unsqueeze(0).expand(batch, -1, -1)
            )
            indices_list.append(
                torch.full(
                    (batch, logits_flat.shape[1]),
                    expert,
                    dtype=torch.long,
                    device=image.device,
                )
            )

        all_logits = torch.cat(logits_list, dim=1)
        all_points = torch.cat(points_list, dim=1)
        all_indices = torch.cat(indices_list, dim=1)
        return {
            "architecture": "native_multiscale",
            "logits": all_logits,
            "points": all_points,
            "base_points": all_points,
            "expert_indices": all_indices,
            "source_expert": all_indices,
        }


@pytest.fixture()
def mini_dataset(tmp_path):
    image_dir = tmp_path / "images" / "val"
    points_dir = tmp_path / "points" / "val"
    image_dir.mkdir(parents=True)
    points_dir.mkdir(parents=True)
    for index in range(2):
        image = np.full(
            (CROP, CROP, 3), 110 + 20 * index, dtype=np.uint8
        )
        cv2.imwrite(
            str(image_dir / f"val_{index}.jpg"),
            image,
        )
        # 每行: 归一化 x y
        with open(points_dir / f"val_{index}.txt", "w") as point_file:
            point_file.write("0.1 0.2\n0.5 0.5\n0.8 0.3\n")
    return tmp_path


@pytest.mark.parametrize(
    ("routing_mode", "expert_index"),
    [("native", None), ("expert_only", 2)],
)
def test_evaluate_native_count_mae_with_fake_model(
    mini_dataset,
    routing_mode,
    expert_index,
):
    """无权重链路覆盖 GT 加载、tile loss 与可视化样本生成。"""
    model = FakePointModel()
    val_dataset = PointDataset(
        str(mini_dataset),
        split="val",
        crop_size=CROP,
    )

    result = evaluate_native_count_mae(
        model,
        val_dataset,
        "cpu",
        criterion=PointMoELoss(),
        crop_size=CROP,
        max_visual_samples=1,
        routing_mode=routing_mode,
        expert_index=expert_index,
    )

    assert model.training
    assert result["mae"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["loss"] is not None
    assert set(result["loss"]) == {"total", "cls", "point", "count"}
    assert result["winner_hist"].shape == (3,)
    assert result["matched_count"] > 0
    assert len(result["validation_samples"]) == 1
    sample = result["validation_samples"][0]
    assert sample["gt_points"].shape == (3, 2)
    assert sample["predictions"]["points"].shape[1] == 2
    assert sample["pred_count"] > 0


@pytest.mark.skipif(
    not os.path.exists("yolo11n.pt"),
    reason="需要本地 yolo11n.pt",
)
def test_evaluate_native_count_mae_expert_only(mini_dataset):
    from models.yolo11_moe_point import YOLO11MoEPoint

    model = YOLO11MoEPoint(
        weights="yolo11n.pt",
        hidden_channels=256,
        native_references=(1, 4, 16),
    )
    val_dataset = PointDataset(
        str(mini_dataset),
        split="val",
        crop_size=CROP,
    )
    result = evaluate_native_count_mae(
        model,
        val_dataset,
        "cpu",
        criterion=PointMoELoss(),
        crop_size=CROP,
        max_visual_samples=1,
        routing_mode="expert_only",
        expert_index=2,
    )
    # 不再 NameError：GT 已加载，计数口径可用。
    assert result["mae"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["bias"] is not None
    assert result["loss"] is not None
    for name in ("total", "cls", "point", "count"):
        assert name in result["loss"]
    assert result["winner_hist"].shape == (3,)
    assert len(result["validation_samples"]) == 1
    sample = result["validation_samples"][0]
    assert sample["gt_points"].shape[1] == 2
    assert sample["pred_count"] > 0


@pytest.mark.skipif(
    not os.path.exists("yolo11n.pt"),
    reason="需要本地 yolo11n.pt",
)
def test_evaluate_native_count_mae_native(mini_dataset):
    from models.yolo11_moe_point import YOLO11MoEPoint

    model = YOLO11MoEPoint(
        weights="yolo11n.pt",
        hidden_channels=256,
        native_references=(1, 4, 16),
    )
    val_dataset = PointDataset(
        str(mini_dataset),
        split="val",
        crop_size=CROP,
    )
    result = evaluate_native_count_mae(
        model,
        val_dataset,
        "cpu",
        criterion=PointMoELoss(),
        crop_size=CROP,
        max_visual_samples=0,
        routing_mode="native",
        expert_index=None,
    )
    assert result["mae"] >= 0.0
    assert result["loss"] is not None
    assert result["validation_samples"] == []
