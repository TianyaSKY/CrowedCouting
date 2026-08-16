"""数据集转换公共逻辑：images + labels(YOLO 虚拟框) + points + dataset.yaml。

供 prepare_jhu / prepare_qnrf / prepare_ucf_cc50 使用，纯标准库。
"""

import os
import shutil


def write_label_files(img_path, points, label_path, point_path):
    """按图片尺寸归一化点坐标，写 YOLO 虚拟框标签与纯点标签。"""
    from ._matlab_utils import jpeg_size

    w, h = jpeg_size(img_path)
    with open(label_path, "w") as fl, open(point_path, "w") as fp:
        for x, y in points:
            nx = max(0.0, min(x / w, 1.0))
            ny = max(0.0, min(y / h, 1.0))
            fl.write(f"0 {nx:.6f} {ny:.6f} 0.010000 0.010000\n")
            fp.write(f"{nx:.6f} {ny:.6f}\n")


def build_dataset(
    dest_dir,
    splits,
    dataset_name=None,
    include_split_in_name=False,
):
    """把 {split: [(img_path, [[x, y], ...]), ...]} 转成标准数据集布局。

    生成 dest_dir/{images,labels,points}/{split}/ 与 dataset.yaml。
    返回 {split: 图片数}。
    """
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.normpath(dest_dir))

    for split in splits:
        for sub in ("images", "labels", "points"):
            os.makedirs(os.path.join(dest_dir, sub, split), exist_ok=True)

    counts = {}
    for split, items in splits.items():
        for i, (img_path, points) in enumerate(items):
            img_name = os.path.basename(img_path)
            if include_split_in_name:
                img_name = f"{split}_{img_name}"
            base = os.path.splitext(img_name)[0]

            shutil.copy(
                img_path, os.path.join(dest_dir, "images", split, img_name)
            )
            write_label_files(
                img_path,
                points,
                os.path.join(dest_dir, "labels", split, base + ".txt"),
                os.path.join(dest_dir, "points", split, base + ".txt"),
            )
            if (i + 1) % 500 == 0:
                print(f"  [{split}] 已处理 {i+1}/{len(items)}")

        counts[split] = len(items)
        print(f"  [{split}] 完成: {len(items)} 张")

    split_names = list(splits)
    train_split = (
        "train"
        if "train" in splits
        else next(
            (name for name in split_names if name.endswith("_train")),
            split_names[0],
        )
    )
    val_split = (
        "val"
        if "val" in splits
        else next(
            (name for name in split_names if name.endswith("_val")),
            split_names[-1],
        )
    )
    test_split = (
        "test"
        if "test" in splits
        else next(
            (name for name in split_names if name.endswith("_test")),
            None,
        )
    )

    with open(os.path.join(dest_dir, "dataset.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(dest_dir)}\n")
        f.write(f"train: images/{train_split}\n")
        f.write(f"val: images/{val_split}\n")
        if test_split is not None:
            f.write(f"test: images/{test_split}\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: person\n")

    print(f"数据集 {dataset_name} 准备完成: {counts}")
    return counts
