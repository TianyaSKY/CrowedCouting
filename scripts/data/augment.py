import os
import glob
import cv2
import random
import numpy as np


def _write_labels_and_points(lbl_path, point_path, entries):
    """同时写出 YOLO 虚拟框标签（5 列）与点标签（2 列）。

    entries: list of (cls_id, nx, ny, box_w, box_h)
    """
    with open(lbl_path, 'w') as f:
        for cls_id, nx, ny, box_w, box_h in entries:
            f.write(
                f"{cls_id} {nx:.6f} {ny:.6f} "
                f"{box_w:.6f} {box_h:.6f}\n"
            )
    with open(point_path, 'w') as f:
        for _, nx, ny, _, _ in entries:
            f.write(f"{nx:.6f} {ny:.6f}\n")


def augment_dataset(dest_dir):
    """
    对合并后的数据集进行离线增强：
    1. 水平翻转 (Horizontal Flip)
    2. 随机区域裁剪 (Random Crop) 并重新计算点相对坐标。
    3. 随机旋转 (Random Rotation) 并重新计算点相对坐标。

    每个增强样本同时写出 labels/（YOLO 5 列虚拟框）与 points/（2 列点），
    Scale-MoE 训练直接读取 points/；v4 分支读取 labels/。
    原图点标签若缺失（尚未运行 prepare_point_labels），需先运行
    `python -m scripts.data.prepare_point_labels` 补齐。
    """
    train_img_dir = os.path.join(dest_dir, 'images', 'train')
    train_lbl_dir = os.path.join(dest_dir, 'labels', 'train')
    train_pt_dir = os.path.join(dest_dir, 'points', 'train')
    os.makedirs(train_pt_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    # 过滤掉已经增强过的图片，防止重复运行导致文件爆炸
    img_paths = [
        p for p in img_paths
        if not any(
            x in os.path.basename(p)
            for x in ['_flip', '_crop', '_rot']
        )
    ]

    print(f"找到 {len(img_paths)} 张原始训练图片，开始进行数据增强...")

    augmented_count = 0
    random.seed(42)  # 固定随机种子确保可复现

    for idx, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        base_name, ext = os.path.splitext(img_name)
        lbl_path = os.path.join(train_lbl_dir, base_name + ".txt")

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 读取原始标签
        labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ])

        # --- 1. 翻转增强 (Horizontal Flip) ---
        flip_img_name = f"{base_name}_flip{ext}"
        flip_img_path = os.path.join(train_img_dir, flip_img_name)
        flip_lbl_path = os.path.join(train_lbl_dir, f"{base_name}_flip.txt")
        flip_pt_path = os.path.join(train_pt_dir, f"{base_name}_flip.txt")

        # 水平翻转图像并保存
        flip_img = cv2.flip(img, 1)
        cv2.imwrite(flip_img_path, flip_img)

        # 翻转点坐标并保存
        flip_entries = []
        for lbl in labels:
            cls_id, nx, ny, box_w, box_h = lbl
            new_nx = 1.0 - nx
            flip_entries.append(
                (cls_id, new_nx, ny, box_w, box_h)
            )
        _write_labels_and_points(
            flip_lbl_path, flip_pt_path, flip_entries
        )
        augmented_count += 1

        # --- 2. 随机裁剪增强 (Random Crop) ---
        # 裁剪 2 张 512x512 区域 (如果图像较小则取图像本身最小边长)
        crop_size = 512
        if h < crop_size or w < crop_size:
            crop_size_h = min(h, crop_size)
            crop_size_w = min(w, crop_size)
        else:
            crop_size_h = crop_size
            crop_size_w = crop_size

        for crop_idx in range(2):
            # 随机选择左上角点
            y1 = random.randint(0, h - crop_size_h)
            x1 = random.randint(0, w - crop_size_w)
            y2 = y1 + crop_size_h
            x2 = x1 + crop_size_w

            crop_img_name = f"{base_name}_crop{crop_idx}{ext}"
            crop_img_path = os.path.join(train_img_dir, crop_img_name)
            crop_lbl_path = os.path.join(
                train_lbl_dir, f"{base_name}_crop{crop_idx}.txt"
            )
            crop_pt_path = os.path.join(
                train_pt_dir, f"{base_name}_crop{crop_idx}.txt"
            )

            # 裁剪图像并保存
            crop_img = img[y1:y2, x1:x2]
            cv2.imwrite(crop_img_path, crop_img)

            # 过滤并重归一化点坐标
            crop_entries = []
            for lbl in labels:
                cls_id, nx, ny, box_w, box_h = lbl
                # 转换回绝对坐标
                px = nx * w
                py = ny * h
                # 判断点是否在裁剪区域内
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # 计算裁剪后的归一化坐标
                    new_nx = (px - x1) / crop_size_w
                    new_ny = (py - y1) / crop_size_h
                    crop_entries.append(
                        (cls_id, new_nx, new_ny, box_w, box_h)
                    )
            _write_labels_and_points(
                crop_lbl_path, crop_pt_path, crop_entries
            )
            augmented_count += 1

        # --- 3. 随机旋转增强 (Random Rotation) ---
        angle = random.uniform(-15.0, 15.0)
        rot_matrix = cv2.getRotationMatrix2D(
            (w / 2.0, h / 2.0), angle, 1.0
        )
        rot_img = cv2.warpAffine(
            img,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        rot_img_name = f"{base_name}_rot{ext}"
        cv2.imwrite(
            os.path.join(train_img_dir, rot_img_name),
            rot_img,
        )

        rot_entries = []
        for lbl in labels:
            cls_id, nx, ny, box_w, box_h = lbl
            px, py = nx * w, ny * h
            rx, ry = rot_matrix @ np.array([px, py, 1.0])
            if 0 <= rx < w and 0 <= ry < h:
                rot_entries.append(
                    (cls_id, rx / w, ry / h, box_w, box_h)
                )
        _write_labels_and_points(
            os.path.join(train_lbl_dir, f"{base_name}_rot.txt"),
            os.path.join(train_pt_dir, f"{base_name}_rot.txt"),
            rot_entries,
        )
        augmented_count += 1

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(img_paths)} 张原始图片的数据增强")

    print(
        f"数据增强全部完成。共新增生成了 {augmented_count} 张训练图片！"
    )


if __name__ == '__main__':
    augment_dataset('datasets/shanghaitech_AB')
