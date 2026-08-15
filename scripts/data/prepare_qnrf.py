"""转换 UCF-QNRF 到项目标准数据集布局（纯标准库）。

源: data/UCF-QNRF_ECCV18/{Train,Test}/（2024 版扁平布局，图片与 *_ann.mat 同目录）
    点标注在 *_ann.mat 的 annPoints 变量（Nx2 double）。
目标: datasets/ucf_qnrf/  {images,labels,points}/{train,test}/ + dataset.yaml

数量: train 1,201 / test 334（共 1,535）。
"""

import glob
import os
import shutil

from ._matlab_utils import mat_points
from ._prepare_common import build_dataset


def prepare_qnrf(data_dir="data", dest_dir="datasets/ucf_qnrf"):
    root = os.path.join(data_dir, "UCF-QNRF_ECCV18")
    if not os.path.isdir(root):
        raise SystemExit(f"未找到源数据: {root}")

    splits = {}
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
        splits[split] = items
        print(f"[{split}] 图片 {len(items)} 张")

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    build_dataset(dest_dir, splits, dataset_name="UCF-QNRF")


if __name__ == "__main__":
    prepare_qnrf()
