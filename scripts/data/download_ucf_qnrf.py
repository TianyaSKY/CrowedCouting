"""下载 UCF-QNRF (Idrees et al. 2018) 人群计数数据集。

官方源: https://www.crcv.ucf.edu/data/ucf-qnrf/
直接下载: UCF-QNRF_ECCV18.zip（约 4.5 GB）

数量情况: 共 1,535 张无约束人群场景图像，约 125 万个人头标注；
训练集 1,201 张，测试集 334 张。

解压后结构（data/UCF-QNRF_ECCV18/）:
  UCF-QNRF_ECCV18/
    Train/img/*.jpg, Train/gt/*.mat
    Test/img/*.jpg,  Test/gt/*.mat
"""

import argparse
import os

from ._download_common import DownloadError, download_file, extract_zip

DEFAULT_URL = "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip"
ZIP_NAME = "UCF-QNRF_ECCV18.zip"
EXPECT_DIR = "UCF-QNRF_ECCV18"


def download_ucf_qnrf(url=None, data_dir="data", keep_zip=False):
    """下载并解压 UCF-QNRF 到 data/UCF-QNRF_ECCV18。"""
    print(
        "UCF-QNRF: 1,535 张图像（train 1,201 / test 334），约 125 万标注；"
        "压缩包约 4.5 GB，请耐心等待"
    )
    url = url or DEFAULT_URL
    os.makedirs(data_dir, exist_ok=True)

    target = os.path.join(data_dir, EXPECT_DIR)
    if os.path.isdir(target):
        print(f"{target} 已存在，跳过下载。")
        return

    zip_path = os.path.join(data_dir, ZIP_NAME)
    if not os.path.exists(zip_path):
        download_file(url, zip_path, note="约 4.5 GB")
    else:
        print(f"已存在 {zip_path}，跳过下载。")

    extract_zip(zip_path, data_dir, expect_tops=(EXPECT_DIR,))

    if not keep_zip and os.path.exists(zip_path):
        os.remove(zip_path)

    if not os.path.isdir(target):
        raise DownloadError(f"解压后未找到 {EXPECT_DIR}/，请检查归档内容")
    print("UCF-QNRF 数据集准备完成。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 UCF-QNRF 人群计数数据集（官方 CRCV 源）"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="覆盖下载地址（镜像源）"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="数据存放目录（默认 data/）"
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="解压后保留 zip 文件"
    )
    return parser.parse_args()


if __name__ == "__main__":
    download_ucf_qnrf(**vars(parse_args()))
