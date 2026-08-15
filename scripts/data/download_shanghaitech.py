"""下载 ShanghaiTech (Zhang et al. 2016) 人群计数数据集 A/B 两部分。

官方仓库 desenzhou/ShanghaiTechDataset 提供的下载源（2024 更新版，Dropbox 镜像，
约 166 MB）: 见 DEFAULT_URL。

数量情况:
  - Part A: 300 train / 182 test（本项目另从 train 随机抽 30 张作 val）
  - Part B: 400 train / 316 test（本项目另从 train 随机抽 40 张作 val）

备用源（需手动获取）:
  - 百度网盘: https://pan.baidu.com/s/1xJnhmJbwPdnNKBM1K6F1Cg?pwd=iga3
  - Kaggle:   https://www.kaggle.com/datasets/xyyu18/shanghaitech-crowd-counting-dataset

解压后结构（data/）:
  part_A_final/{train,test}_data/images/*.jpg, ground_truth/*.mat
  part_B_final/{train,test}_data/images/*.jpg, ground_truth/*.mat
"""

import argparse
import os

from ._download_common import DownloadError, download_file, extract_zip

DEFAULT_URL = (
    "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/"
    "ShanghaiTech_Crowd_Counting_Dataset.zip"
    "?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
)
ZIP_NAME = "ShanghaiTech_Crowd_Counting_Dataset.zip"
EXPECT_DIRS = ("part_A_final", "part_B_final")


def download_shanghaitech(url=None, data_dir="data", keep_zip=False):
    """下载并解压 ShanghaiTech 数据集到 data/part_A_final 与 data/part_B_final。"""
    print(
        "ShanghaiTech: Part A 300 train / 182 test，Part B 400 train / 316 test；"
        "压缩包约 166 MB"
    )
    url = url or DEFAULT_URL
    os.makedirs(data_dir, exist_ok=True)

    # 已存在则跳过
    if all(
        os.path.isdir(os.path.join(data_dir, name))
        for name in EXPECT_DIRS
    ):
        print("data/part_A_final 与 data/part_B_final 已存在，跳过下载。")
        return

    zip_path = os.path.join(data_dir, ZIP_NAME)

    if not os.path.exists(zip_path):
        download_file(url, zip_path, note="约 166 MB")
    else:
        print(f"已存在 {zip_path}，跳过下载。")

    extract_zip(zip_path, data_dir, expect_tops=set(EXPECT_DIRS))

    if not keep_zip and os.path.exists(zip_path):
        os.remove(zip_path)

    if not all(
        os.path.isdir(os.path.join(data_dir, name))
        for name in EXPECT_DIRS
    ):
        raise DownloadError("解压后未找到 part_A_final/part_B_final，请检查镜像源内容")
    print("ShanghaiTech 数据集准备完成。")
    print("下一步: python -m scripts.data.prepare_combined")


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 ShanghaiTech 人群计数数据集（官方 Dropbox 镜像）"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="覆盖下载地址（百度网盘/Kaggle 等镜像不可直接脚本下载时手动转链）"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="数据存放目录（默认 data/，与 prepare_combined 一致）"
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="解压后保留 zip 文件"
    )
    return parser.parse_args()


if __name__ == "__main__":
    download_shanghaitech(**vars(parse_args()))
