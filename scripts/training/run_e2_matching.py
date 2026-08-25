"""E2 matching starvation 诊断实验：confidence 预筛 vs 全池 Hungarian。

对照组与实验组保持 seed/数据/epoch/LR 完全一致，仅改两个 matching 参数：

- baseline:   --match-top-k 2000 --match-confidence-weight 0.25
              （当前默认：先从 6400 个 E2 candidate 按 confidence 预筛
              2000 个，再 Hungarian；代价含 confidence 项）
- full_pool:  --match-top-k 6400 --match-confidence-weight 0.0
              （取消 confidence 预筛与置信度项：全部 6400 个 E2
              candidate 参与纯位置 Hungarian）

评估对比先看 scale-bin 的 large-proxy Recall@16/32、matched distance、
matched confidence、count bias，而不是只看 MAE：

    python -m scripts.evaluation.ablation_expert \
        --checkpoint runs/e2_matching_baseline/best_native.pt \
        --checkpoint runs/e2_matching_full_pool/best_native.pt \
        --data-root datasets/shanghaitech_AB

训练前应先跑 smoke test（1 样本、1 epoch、batch=1），确认
forward -> loss -> backward -> optimizer step -> checkpoint 保存
-> reload 全链路正常。
"""
from __future__ import annotations

import subprocess
import sys

COMMON = [
    "python", "-m", "scripts.training.train_moe",
    "--weights", "yolo11n.pt",
    "--data-root", "datasets/shanghaitech_AB",
    "--crop-size", "640",
    "--batch-size", "8",
    "--epochs", "100",
    "--hidden-channels", "256",
    "--native-references", "1,4,16",
    "--native-warmup-epochs", "5",
    "--backbone-lr", "1e-4",
    "--head-lr", "1e-3",
    "--weight-decay", "1e-4",
    "--grad-clip", "10",
    "--workers", "4",
    "--val-image-interval", "1",
    "--val-image-count", "4",
    "--val-image-conf", "0.5",
    "--seed", "2026",
    "--expert-index", "2",
    "--freeze-epochs", "3",
]

RUNS = [
    (
        "e2_matching_baseline",
        ["--match-top-k", "2000", "--match-confidence-weight", "0.25"],
    ),
    (
        "e2_matching_full_pool",
        ["--match-top-k", "6400", "--match-confidence-weight", "0.0"],
    ),
]


def main() -> int:
    print("E2_MATCHING_DRIVER_STARTED", flush=True)
    failures = 0
    for name, extra in RUNS:
        command = COMMON + extra + ["--save-dir", f"runs/{name}"]
        print(f"\n===== RUN {name} =====", flush=True)
        result = subprocess.run(command, cwd=".")
        if result.returncode != 0:
            failures += 1
            print(f"RUN {name} FAILED rc={result.returncode}", flush=True)
        else:
            print(f"RUN {name} OK", flush=True)
    print(f"E2_MATCHING_DRIVER_DONE failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
