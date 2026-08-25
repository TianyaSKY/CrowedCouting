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
