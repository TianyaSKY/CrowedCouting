import glob
import os


def convert_split_labels(label_dir, out_dir):
    """将 YOLO 格式的虚拟框标签（cls nx ny w h）转换为点标签（nx ny）。

    现有 prepare_combined.py 生成的是 YOLO 格式虚拟框标签，中心点即人头位置。
    本脚本将其转换为 PointDataset 需要的纯点标注格式：
        每行: x_normalized y_normalized
    """
    os.makedirs(out_dir, exist_ok=True)

    label_paths = sorted(
        glob.glob(os.path.join(label_dir, "*.txt"))
    )

    print(f"找到 {len(label_paths)} 个标签文件，转换到 {out_dir}")

    for label_path in label_paths:
        base_name = os.path.basename(label_path)
        out_path = os.path.join(out_dir, base_name)

        lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    # YOLO 格式: cls nx ny w h -> 只取归一化中心点
                    lines.append(f"{parts[1]} {parts[2]}\n")

        with open(out_path, "w") as f:
            f.writelines(lines)

    print(f"转换完成，共写入 {len(label_paths)} 个点标签文件")


def prepare_point_labels(dest_dir="datasets/shanghaitech_AB"):
    """为训练集和验证集分别转换点标签。"""
    for split in ("train", "val"):
        label_dir = os.path.join(dest_dir, "labels", split)
        out_dir = os.path.join(dest_dir, "points", split)

        if not os.path.isdir(label_dir):
            print(f"跳过 {label_dir}: 目录不存在")
            continue

        convert_split_labels(label_dir, out_dir)


if __name__ == "__main__":
    prepare_point_labels()
