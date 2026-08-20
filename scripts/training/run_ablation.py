"""Run the single-expert ablation suite sequentially on one GPU.

实验 1 (冻结 YOLO Backbone+Neck 全程): E0 / E1 / E2
实验 2 (前 3 epoch 冻结后微调):        E0 / E1 / E2
实验 3 (联合 competitive, 同种子复跑): native

全部 run 共用: 同 pretrained 权重、同 train/val split、100 epochs、
同增广(经 --seed 固定)、同 optimizer/LR、同 matching cost。
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
    "--match-top-k", "2000",
    "--match-position-weight", "5.0",
    "--match-confidence-weight", "0.25",
    "--val-image-interval", "1",
    "--val-image-count", "4",
    "--val-image-conf", "0.5",
    "--seed", "2026",
]

RUNS = [
    ("ablation_frozen_E0", ["--expert-index", "0", "--freeze-epochs", "100"]),
    ("ablation_frozen_E1", ["--expert-index", "1", "--freeze-epochs", "100"]),
    ("ablation_frozen_E2", ["--expert-index", "2", "--freeze-epochs", "100"]),
    ("ablation_ft_E0", ["--expert-index", "0", "--freeze-epochs", "3"]),
    ("ablation_ft_E1", ["--expert-index", "1", "--freeze-epochs", "3"]),
    ("ablation_ft_E2", ["--expert-index", "2", "--freeze-epochs", "3"]),
    ("ablation_joint", ["--freeze-epochs", "3"]),
]


def main() -> int:
    print("ABLATION_DRIVER_STARTED", flush=True)
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
    print(f"ABLATION_DRIVER_DONE failures={failures}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
