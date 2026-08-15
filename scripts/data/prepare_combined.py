"""合并 ShanghaiTech Part A/B 生成 YOLO 虚拟框标签（纯标准库，无需 scipy/h5py/cv2）。

输出 datasets/shanghaitech_AB/:
    images/{train,val}/*.jpg    图片（part_A_/part_B_ 前缀，val = 原始 test_data）
    labels/{train,val}/*.txt    YOLO 虚拟框: 0 nx ny 0.010000 0.010000
    dataset.yaml                ultralytics 数据配置
"""

import glob
import os
import shutil

from ._matlab_utils import jpeg_size, mat_points


def _write_yaml(path, dest):
    """手写 ultralytics dataset.yaml（避免依赖 PyYAML）。"""
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(dest)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: person\n")


def _process_single_part(src_dir, dest_dir, prefix):
    """处理单个 Part 并在文件名前加上前缀，防止冲突。"""
    for split in ("train", "val"):
        os.makedirs(os.path.join(dest_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split), exist_ok=True)

    train_images = sorted(
        glob.glob(os.path.join(src_dir, "train_data", "images", "*.jpg"))
    )
    test_images = sorted(
        glob.glob(os.path.join(src_dir, "test_data", "images", "*.jpg"))
    )

    splits = {"train": train_images, "val": test_images}

    for split, img_list in splits.items():
        for i, img_path in enumerate(img_list):
            img_name = os.path.basename(img_path)
            new_img_name = f"{prefix}_{img_name}"

            dest_img_path = os.path.join(
                dest_dir, "images", split, new_img_name
            )
            shutil.copy(img_path, dest_img_path)

            mat_name = "GT_" + img_name.replace(".jpg", ".mat")
            if split == "train":
                mat_path = os.path.join(
                    src_dir, "train_data", "ground_truth", mat_name
                )
            else:
                mat_path = os.path.join(
                    src_dir, "test_data", "ground_truth", mat_name
                )

            points = mat_points(mat_path)

            w, h = jpeg_size(img_path)

            new_label_name = new_img_name.replace(".jpg", ".txt")
            dest_label_path = os.path.join(
                dest_dir, "labels", split, new_label_name
            )

            with open(dest_label_path, "w") as f:
                for x, y in points:
                    nx = max(0.0, min(x / w, 1.0))
                    ny = max(0.0, min(y / h, 1.0))
                    f.write(f"0 {nx:.6f} {ny:.6f} 0.010000 0.010000\n")

            if (i + 1) % 100 == 0:
                print(f"[{prefix}] 已处理 {i+1}/{len(img_list)} 张 {split} 图片")


def prepare_combined_dataset():
    dest = "datasets/shanghaitech_AB"
    print(f"正在清理并准备合并后的数据集文件夹: {dest}")
    if os.path.exists(dest):
        shutil.rmtree(dest)

    _process_single_part("data/part_A_final", dest, "part_A")
    _process_single_part("data/part_B_final", dest, "part_B")

    _write_yaml(os.path.join(dest, "dataset.yaml"), dest)
    print(f"合并数据集完成。YAML 配置文件已保存至 {dest}/dataset.yaml")


if __name__ == "__main__":
    prepare_combined_dataset()
