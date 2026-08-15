"""转换 JHU-Crowd++ 到项目标准数据集布局（纯标准库）。

源: data/jhu_crowd_v2.0/{train,val,test}/{images,gt}
    社区镜像的 gt 为逐图 txt（每行: x y w h flag flag），取前两列 (x, y) 作为人头中心。
目标: datasets/jhu_crowd/  {images,labels,points}/{train,val,test}/ + dataset.yaml

数量: train 2,272 / val 500 / test 1,600（共 4,372）。
"""

import glob
import os
import shutil

from ._prepare_common import build_dataset


def parse_gt_txt(gt_path):
    """读取 JHU txt 点标注（每行 x y ...），返回 [[x, y], ...]。"""
    points = []
    with open(gt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])
    return points


def prepare_jhu(data_dir="data", dest_dir="datasets/jhu_crowd"):
    root = os.path.join(data_dir, "jhu_crowd_v2.0")
    if not os.path.isdir(root):
        raise SystemExit(f"未找到源数据: {root}（先运行 download_jhu_crowd）")

    splits = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(root, split, "images")
        gt_dir = os.path.join(root, split, "gt")
        if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
            print(f"跳过 {split}: 目录缺失")
            continue

        items = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_dir, base + ".txt")
            if os.path.exists(gt_path):
                items.append((img_path, parse_gt_txt(gt_path)))
        splits[split] = items
        print(f"[{split}] 图片 {len(items)} 张")

    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    build_dataset(dest_dir, splits, dataset_name="JHU-Crowd++")


if __name__ == "__main__":
    prepare_jhu()
