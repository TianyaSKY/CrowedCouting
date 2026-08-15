"""下载 UCF-CC-50 (Idrees et al. 2013) 人群计数数据集。

官方源: https://www.crcv.ucf.edu/data/ucf-cc-50/
直接下载: UCFCrowdCountingDataset_CVPR13.rar（约 8 MB，RAR 格式）

数量情况: 共 50 张图像（极端密集人群），每张附 *_ann.mat 点标注。
按 Idrees et al. 2013 标准协议做 5 折交叉验证（每折 40 train / 10 test）。

解压后结构（data/UCF_CC_50/）:
  UCF_CC_50/
    Readme.txt
    1.jpg ... 50.jpg
    1_ann.mat ... 50_ann.mat
"""

import argparse
import os

from ._download_common import DownloadError, download_file, extract_rar

DEFAULT_URL = "https://www.crcv.ucf.edu/data/ucf-cc-50/UCFCrowdCountingDataset_CVPR13.rar"
RAR_NAME = "UCF_CC_50.rar"
EXPECT_DIR = "UCF_CC_50"


def download_ucf_cc50(url=None, data_dir="data", keep_rar=False):
    """下载并解压 UCF-CC-50 到 data/UCF_CC_50。"""
    print(
        "UCF-CC-50: 50 张极端密集人群图像 + 点标注（5 折交叉验证协议，"
        "每折 40 train / 10 test）"
    )
    url = url or DEFAULT_URL
    os.makedirs(data_dir, exist_ok=True)

    target = os.path.join(data_dir, EXPECT_DIR)
    if os.path.isdir(target):
        print(f"{target} 已存在，跳过下载。")
        return

    rar_path = os.path.join(data_dir, RAR_NAME)
    if not os.path.exists(rar_path):
        download_file(url, rar_path, note="约 8 MB")
    else:
        print(f"已存在 {rar_path}，跳过下载。")

    extract_rar(rar_path, data_dir)

    if not keep_rar and os.path.exists(rar_path):
        os.remove(rar_path)

    if not os.path.isdir(target):
        raise DownloadError(
            f"解压后未找到 {EXPECT_DIR}/，请检查归档内容"
        )
    print("UCF-CC-50 数据集准备完成。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 UCF-CC-50 人群计数数据集（官方 CRCV 源）"
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
        "--keep-rar", action="store_true",
        help="解压后保留 rar 文件"
    )
    return parser.parse_args()


if __name__ == "__main__":
    download_ucf_cc50(**vars(parse_args()))
