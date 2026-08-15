"""转换 UCF-CC-50 到项目标准数据集布局，并按 Idrees et al. 2013 协议划分 5 折。

源: data/UCF_CC_50/{1..50}.jpg + {1..50}_ann.mat（annPoints, Nx2）
目标: datasets/ucf_cc50/  {images,labels,points}/{fold{i}_train,fold{i}_test}/ + dataset.yaml

划分: 50 张图随机打乱（固定 seed=0，可 --seed 覆盖）后等分 5 折，
每折 10 张 test / 40 张 train，训练第 i 折时用 fold{i}_train + fold{i}_test。
"""

import argparse
import glob
import os
import random
import shutil

from ._matlab_utils import mat_points
from ._prepare_common import build_dataset


def prepare_ucf_cc50(data_dir="data", dest_dir="datasets/ucf_cc50", seed=0):
    root = os.path.join(data_dir, "UCF_CC_50")
    if not os.path.isdir(root):
        raise SystemExit(f"未找到源数据: {root}")

    items = []
    for img_path in sorted(glob.glob(os.path.join(root, "*.jpg"))):
        base = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(root, base + "_ann.mat")
        if os.path.exists(mat_path):
            items.append((img_path, mat_points(mat_path)))
    print(f"共 {len(items)} 张图")

    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)

    splits = {}
    for fold in range(5):
        test_ids = order[fold * 10 : (fold + 1) * 10]
        train_ids = [i for i in order if i not in set(test_ids)]
        splits[f"fold{fold}_train"] = [items[i] for i in train_ids]
        splits[f"fold{fold}_test"] = [items[i] for i in test_ids]
        print(
            f"fold{fold}: train {len(train_ids)} / test {len(test_ids)}"
        )

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    build_dataset(dest_dir, splits, dataset_name="UCF-CC-50")


def parse_args():
    parser = argparse.ArgumentParser(
        description="转换 UCF-CC-50 并生成 5 折交叉验证划分"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dest-dir", type=str, default="datasets/ucf_cc50")
    parser.add_argument("--seed", type=int, default=0,
                        help="划分随机种子（默认 0）")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_ucf_cc50(**vars(parse_args()))
