"""转换 UCF-QNRF 到项目标准数据集布局（纯标准库）。

源: data/UCF-QNRF_ECCV18/{Train,Test}/（2024 版扁平布局，图片与 *_ann.mat 同目录）
    点标注在 *_ann.mat 的 annPoints 变量（Nx2 double）。
目标: datasets/ucf_qnrf/  {images,labels,points}/{train,val,test}/ + dataset.yaml

数量: 官方 Train 1,201 拆为 train 约 1,081 / val 约 120；官方 Test 334
原样保留为 test（共 1,535）。
"""

import argparse
import glob
import os
import random
import shutil

from ._matlab_utils import mat_points
from ._prepare_common import build_dataset


def prepare_qnrf(data_dir="data", dest_dir="datasets/ucf_qnrf", seed=0):
    root = os.path.join(data_dir, "UCF-QNRF_ECCV18")
    if not os.path.isdir(root):
        raise SystemExit(f"未找到源数据: {root}")

    source_items = {}
    for split, src_name in (("train", "Train"), ("test", "Test")):
        src_dir = os.path.join(root, src_name)
        if not os.path.isdir(src_dir):
            print(f"跳过 {split}: 目录缺失 {src_dir}")
            continue

        items = []
        for img_path in sorted(glob.glob(os.path.join(src_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            mat_path = os.path.join(src_dir, base + "_ann.mat")
            if os.path.exists(mat_path):
                items.append((img_path, mat_points(mat_path)))
        source_items[split] = items
        print(f"[{split}] 图片 {len(items)} 张")

    train_items = list(source_items.get("train", []))
    test_items = list(source_items.get("test", []))
    rng = random.Random(seed)
    rng.shuffle(train_items)
    val_size = max(1, round(len(train_items) * 0.1))

    splits = {
        "train": train_items[val_size:],
        "val": train_items[:val_size],
        "test": test_items,
    }
    print(
        f"[split] train {len(splits['train'])} / "
        f"val {len(splits['val'])} / test {len(splits['test'])} "
        f"(seed={seed})"
    )

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    build_dataset(
        dest_dir,
        splits,
        dataset_name="UCF-QNRF",
        include_split_in_name=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="转换 UCF-QNRF 并从官方 Train 固定切分 train/val"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dest-dir", type=str, default="datasets/ucf_qnrf")
    parser.add_argument("--seed", type=int, default=0)
    prepare_qnrf(**vars(parser.parse_args()))
