from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_points(point_path: str, width: int, height: int) -> np.ndarray:
    """读取点标注文件（每行: x_normalized y_normalized），返回像素坐标 [N, 2]。"""
    points = []
    if os.path.exists(point_path):
        with open(point_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                nx, ny = float(parts[0]), float(parts[1])
                points.append([nx * width, ny * height])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


class PointDataset(Dataset):
    """点监督人群定位数据集。

    目录结构：
        root/
        ├── images/
        │   ├── train/*.jpg
        │   └── val/*.jpg
        └── points/
            ├── train/*.txt     # 每行: x_normalized y_normalized
            └── val/*.txt

    输出：
        "img":    Tensor[3, H, W]（RGB，0~1，当前图像像素坐标空间）
        "points": Tensor[N, 2]（像素坐标）
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 640,
        augment: bool = True,
    ) -> None:
        super().__init__()

        self.image_dir = os.path.join(root, "images", split)
        self.points_dir = os.path.join(root, "points", split)
        self.crop_size = crop_size
        self.augment = augment

        self.image_paths = sorted(
            glob.glob(os.path.join(self.image_dir, "*.jpg"))
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"未在 {self.image_dir} 中找到任何 jpg 图片"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def sample_gt_counts(
        self,
        num_samples: int = 100,
    ) -> np.ndarray:
        """随机采样若干样本，统计裁剪/增强后的每图 GT 点数分布。

        用于诊断随机 crop 产生过多空样本（GT=0）的问题。
        """
        sample_size = min(num_samples, len(self))
        indices = random.sample(range(len(self)), sample_size)

        counts = np.empty(sample_size, dtype=np.int64)
        for i, index in enumerate(indices):
            counts[i] = self[index]["points"].shape[0]

        return counts

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        point_path = os.path.join(
            self.points_dir, base_name + ".txt"
        )

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图片 {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        points = load_points(point_path, width, height)

        if self.augment:
            image, points = self._augment_train(
                image, points, height, width
            )
        else:
            # 验证/测试：letterbox 保持纵横比，避免压扁改变人的尺度
            image, points = self._letterbox(
                image, points, height, width
            )

        image_tensor = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        points_tensor = torch.from_numpy(points)

        return {
            "img": image_tensor,
            "points": points_tensor,
        }

    def _augment_train(
        self,
        image: np.ndarray,
        points: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        crop_size = self.crop_size

        # 1. 轻度缩放（不改变纵横比）
        scale = random.uniform(0.8, 1.2)
        if abs(scale - 1.0) > 1e-3:
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            points = points * np.array(
                [new_width / width, new_height / height],
                dtype=np.float32,
            )
            height, width = new_height, new_width

        # 2. 随机裁剪到 crop_size x crop_size
        if height >= crop_size and width >= crop_size:
            y0 = random.randint(0, height - crop_size)
            x0 = random.randint(0, width - crop_size)
            image = image[y0:y0 + crop_size, x0:x0 + crop_size]
            points = points - np.array(
                [x0, y0], dtype=np.float32
            )
        else:
            # 图像小于裁剪尺寸时直接放大
            image = cv2.resize(
                image,
                (crop_size, crop_size),
                interpolation=cv2.INTER_LINEAR,
            )
            fx = crop_size / max(width, 1)
            fy = crop_size / max(height, 1)
            points = points * np.array(
                [fx, fy], dtype=np.float32
            )

        # 3. 保留裁剪区域内的点
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < crop_size)
            & (points[:, 1] >= 0)
            & (points[:, 1] < crop_size)
        )
        points = points[keep]

        if points.shape[0] > 0:
            points[:, 0] = np.clip(points[:, 0], 0, crop_size - 1)
            points[:, 1] = np.clip(points[:, 1], 0, crop_size - 1)

        # 4. 水平翻转
        if random.random() < 0.5:
            image = image[:, ::-1]
            if points.shape[0] > 0:
                points[:, 0] = crop_size - 1 - points[:, 0]

        # 5. 亮度与对比度变化
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20.0, 20.0)
        image = cv2.convertScaleAbs(
            image, alpha=alpha, beta=beta
        )

        return image, points

    def _letterbox(
        self,
        image: np.ndarray,
        points: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """等比例缩放后居中填充到 crop_size x crop_size，同步变换点坐标。

        对尺度敏感的人群计数任务，直接压成正方形会改变人的纵横比与
        尺度分布，这里保持原始宽高比。
        """
        crop_size = self.crop_size

        scale = min(crop_size / max(width, 1),
                    crop_size / max(height, 1))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (crop_size - new_width) // 2
        pad_y = (crop_size - new_height) // 2

        image = cv2.copyMakeBorder(
            image,
            pad_y,
            crop_size - new_height - pad_y,
            pad_x,
            crop_size - new_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        if points.shape[0] > 0:
            points = points * scale
            points = points + np.array(
                [pad_x, pad_y], dtype=np.float32
            )
            points[:, 0] = np.clip(
                points[:, 0], 0, crop_size - 1
            )
            points[:, 1] = np.clip(
                points[:, 1], 0, crop_size - 1
            )

        return image, points


def point_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, object]:
    """由于每张图人数不同，points 保留为列表。"""
    images = torch.stack(
        [sample["img"] for sample in batch]
    )

    points = [
        sample["points"]
        for sample in batch
    ]

    return {
        "img": images,
        "points": points,
    }
