"""一个命令转换全部数据集到标准布局（幂等，纯标准库）。

用法:
    python -m scripts.data.prepare_all            # 转换全部（已存在则跳过）
    python -m scripts.data.prepare_all --force    # 全部重新转换

产出（datasets/ 下）:
    shanghaitech_AB  train 630 / val 70 / test 498
    jhu_crowd        train 2,272 / val 500 / test 1,600
    ucf_qnrf         train 约 1,081 / val 约 120 / test 334
    ucf_cc50         5 折 × fold{i}_train(36) / fold{i}_val(4) /
                     fold{i}_test(10)
"""

import argparse
import os
import shutil

from .prepare_combined import prepare_combined_dataset
from .prepare_jhu import prepare_jhu
from .prepare_point_labels import prepare_point_labels
from .prepare_qnrf import prepare_qnrf
from .prepare_ucf_cc50 import prepare_ucf_cc50

STEPS = [
    ("ShanghaiTech A+B",
     "datasets/shanghaitech_AB",
     lambda: prepare_combined_dataset()),
    ("ShanghaiTech 点标签",
     "datasets/shanghaitech_AB/points",
     lambda: prepare_point_labels()),
    ("JHU-Crowd++",
     "datasets/jhu_crowd",
     lambda: prepare_jhu()),
    ("UCF-QNRF",
     "datasets/ucf_qnrf",
     lambda: prepare_qnrf()),
    ("UCF-CC-50 (5 折)",
     "datasets/ucf_cc50",
     lambda: prepare_ucf_cc50()),
]


def prepare_all(force=False):
    for name, marker, fn in STEPS:
        if os.path.isdir(marker) and not force:
            print(f"[跳过] {name}: {marker} 已存在（--force 重新转换）")
            continue
        print(f"\n===== {name} =====")
        fn()
    print("\n全部数据集转换完成。\n下一步: python -m scripts.training.train_all")


def parse_args():
    parser = argparse.ArgumentParser(
        description="一个命令转换全部数据集到标准布局"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="已存在的数据集也重新转换（会删除重建目标目录）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare_all(**vars(parse_args()))
