from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def letterbox_image(
    image: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, float, int, int]:
    """Resize an image with aspect-ratio preservation and centered padding."""
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    height, width = image.shape[:2]
    scale = min(
        crop_size / max(width, 1),
        crop_size / max(height, 1),
    )
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_x = (crop_size - new_width) // 2
    pad_y = (crop_size - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        crop_size - new_height - pad_y,
        pad_x,
        crop_size - new_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, pad_x, pad_y


def letterbox_points(
    points: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    crop_size: int,
) -> np.ndarray:
    """Map original-image points into letterboxed pixel coordinates."""
    transformed = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if transformed.shape[0] == 0:
        return transformed
    transformed *= np.asarray([scale, scale], dtype=np.float32)
    transformed += np.asarray([pad_x, pad_y], dtype=np.float32)
    transformed[:, 0] = np.clip(transformed[:, 0], 0, crop_size - 1)
    transformed[:, 1] = np.clip(transformed[:, 1], 0, crop_size - 1)
    return transformed


def inverse_letterbox_points(
    points: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Map letterboxed pixel coordinates back into original-image space."""
    if scale <= 0:
        raise ValueError("letterbox scale must be positive")
    transformed = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if transformed.shape[0] == 0:
        return transformed
    transformed -= np.asarray([pad_x, pad_y], dtype=np.float32)
    transformed /= scale
    transformed[:, 0] = np.clip(transformed[:, 0], 0, max(width - 1, 0))
    transformed[:, 1] = np.clip(transformed[:, 1], 0, max(height - 1, 0))
    return transformed



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
        "image_path": 原始图片的稳定绝对/相对路径字符串

    训练在线增强（augment=True）：
        缩放 0.8–1.2 → 随机裁剪 crop_size → 水平翻转 p=0.5
        → 随机旋转 ±10° p=0.5 → 亮度/对比度 → HSV 色相/饱和度抖动
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

    def __getitem__(self, index: int) -> dict[str, object]:
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
            "image_path": image_path,
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
        else:
            # 小图先等比例放大到短边至少为 crop_size，再随机裁剪。
            # 禁止直接拉伸成正方形，否则 x/y 像素尺度会被分别改变，
            # 进而污染点间距与 Router 尺度标签。
            scale_up = max(
                crop_size / max(height, 1),
                crop_size / max(width, 1),
            )
            new_height = max(
                crop_size,
                int(round(height * scale_up)),
            )
            new_width = max(
                crop_size,
                int(round(width * scale_up)),
            )
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            points = points * np.array(
                [new_width / max(width, 1),
                 new_height / max(height, 1)],
                dtype=np.float32,
            )
            height, width = new_height, new_width
            y0 = random.randint(0, height - crop_size)
            x0 = random.randint(0, width - crop_size)

        image = image[y0:y0 + crop_size, x0:x0 + crop_size]
        points = points - np.array(
            [x0, y0], dtype=np.float32
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

        # 5. 随机旋转（小角度，保持头部近似圆形；点与图像用同一
        #    旋转矩阵同步变换，旋转出界点直接丢弃）
        if random.random() < 0.5:
            angle = random.uniform(-10.0, 10.0)
            rotation_matrix = cv2.getRotationMatrix2D(
                (crop_size / 2.0, crop_size / 2.0),
                angle,
                1.0,
            ).astype(np.float32)
            image = cv2.warpAffine(
                image,
                rotation_matrix,
                (crop_size, crop_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            if points.shape[0] > 0:
                ones = np.ones(
                    (points.shape[0], 1),
                    dtype=np.float32,
                )
                points = (
                    rotation_matrix
                    @ np.hstack([points, ones]).T
                ).T
                keep = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < crop_size)
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < crop_size)
                )
                points = points[keep]
                if points.shape[0] > 0:
                    points[:, 0] = np.clip(
                        points[:, 0], 0, crop_size - 1
                    )
                    points[:, 1] = np.clip(
                        points[:, 1], 0, crop_size - 1
                    )

        # 6. 亮度与对比度变化
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20.0, 20.0)
        image = cv2.convertScaleAbs(
            image, alpha=alpha, beta=beta
        )

        # 7. 色相/饱和度抖动（HSV 空间；亮度已由第 6 步控制）
        if random.random() < 0.5:
            image_hsv = cv2.cvtColor(
                image, cv2.COLOR_RGB2HSV
            ).astype(np.float32)
            hue_delta = random.uniform(-5.0, 5.0)
            saturation_scale = random.uniform(0.85, 1.15)
            image_hsv[..., 0] = (
                image_hsv[..., 0] + hue_delta
            ) % 180.0
            image_hsv[..., 1] = np.clip(
                image_hsv[..., 1] * saturation_scale,
                0.0,
                255.0,
            )
            image = cv2.cvtColor(
                image_hsv.astype(np.uint8),
                cv2.COLOR_HSV2RGB,
            )

        return image, points

    def _letterbox(
        self,
        image: np.ndarray,
        points: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """等比例缩放后居中填充到 crop_size x crop_size，同步变换点坐标。"""
        padded, scale, pad_x, pad_y = letterbox_image(
            image,
            self.crop_size,
        )
        return padded, letterbox_points(
            points,
            scale,
            pad_x,
            pad_y,
            self.crop_size,
        )


def point_collate_fn(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    """由于每张图人数不同，points 保留为列表并保留图片路径。"""
    images = torch.stack(
        [sample["img"] for sample in batch]  # type: ignore[arg-type]
    )

    points = [
        sample["points"] for sample in batch
    ]

    return {
        "img": images,
        "points": points,
        "image_paths": [
            str(sample["image_path"]) for sample in batch
        ],
    }


