"""Tests for scale-bin metrics (Commit B: large-target diagnosis)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluation.ablation_expert import (
    SCALE_BIN_NAMES,
    ScaleBinStats,
    assign_scale_bin,
    scale_bin_thresholds,
    scale_proxies,
)


def test_scale_proxies_dense_vs_isolated():
    # 三丛密集 + 一个孤立点：孤立点 proxy 必须远大于密集点。
    gt = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [500.0, 500.0],
        ],
        dtype=np.float32,
    )
    proxies = scale_proxies(gt)
    assert proxies[4] > 10 * proxies[0]
    assert np.all(np.isfinite(proxies))


def test_scale_proxies_single_gt_is_inf():
    gt = np.array([[10.0, 20.0]], dtype=np.float32)
    proxies = scale_proxies(gt)
    assert proxies.shape == (1,)
    assert np.isinf(proxies[0])


def test_scale_proxies_two_gt():
    gt = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    proxies = scale_proxies(gt)
    assert np.allclose(proxies, 5.0)


def test_thresholds_and_binning():
    proxies = np.array([1.0, 2.0, 3.0, 100.0, 200.0, np.inf])
    low, high = scale_bin_thresholds(proxies)
    assert low < high
    bins = [assign_scale_bin(float(p), (low, high)) for p in proxies]
    assert bins[:2] == ["dense_small", "dense_small"]
    assert bins[2] == "medium"
    assert bins[3:5] == ["sparse_large", "sparse_large"]
    # inf 归入 sparse_large
    assert bins[5] == "sparse_large"


def test_scale_bin_stats_recall_and_neighborhood():
    stats = ScaleBinStats()
    # 命中 @8、命中 @32 但未命中 @16、完全未命中（inf）
    stats.update(4.0, 0.9, 0.95)
    stats.update(20.0, 0.5, 0.8)
    stats.update(np.inf, 0.0, 0.0)
    summary = stats.summary()
    assert summary["gt_total"] == 3
    assert summary["recall@8px"] == pytest.approx(1.0 / 3.0)
    assert summary["recall@16px"] == pytest.approx(1.0 / 3.0)
    assert summary["recall@32px"] == pytest.approx(2.0 / 3.0)
    # mean/median 只统计有限距离
    assert summary["mean_dist_px"] == pytest.approx(12.0)
    assert summary["median_dist_px"] == pytest.approx(12.0)
    assert summary["raw_max_conf@16px"] == pytest.approx(
        (0.9 + 0.5 + 0.0) / 3.0
    )
    assert summary["raw_max_conf@32px"] == pytest.approx(
        (0.95 + 0.8 + 0.0) / 3.0
    )


def test_scale_bin_names_consistency():
    assert SCALE_BIN_NAMES == ("dense_small", "medium", "sparse_large")
