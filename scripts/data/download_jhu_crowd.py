"""下载 JHU-Crowd++ (Sindagi et al. 2019) 人群计数数据集。

官方源: http://www.crowd-counting.com/（Google Drive 托管，页面需填表同意条款）。
本脚本直接下载社区公开镜像目录（jhucrowdv2.0）中的 jhu_crowd_v2.0.zip（约 2.87 GB）:
  目录: https://drive.google.com/drive/folders/1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n
  文件: https://drive.google.com/file/d/1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj

数量情况: 共 4,372 张图像（train 2,272 / val 500 / test 1,600），约 151 万个人头标注；
单图人数最高 25,791，包含下雪/下雨/雾霾等恶劣天气场景及无人图像。

解压后结构（data/jhu_crowd_v2.0/）:
  jhu_crowd_v2.0/
    train/images/*.jpg, train/gt/*.mat
    val/images/*.jpg,   val/gt/*.mat
    test/images/*.jpg,  test/gt/*.mat
"""

import argparse
import os
import re
import urllib.request

from ._download_common import (
    DownloadError,
    USER_AGENT,
    build_opener,
    extract_zip,
    open_url,
)

DRIVE_FILE_ID = "1pA7ZeXU3hh-1txS9lFQiCek1ts3MdBaj"  # jhu_crowd_v2.0.zip
DRIVE_FOLDER_ID = "1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n"  # 镜像目录
ZIP_NAME = "jhu_crowd_v2.0.zip"
EXPECT_DIR = "jhu_crowd_v2.0"


def _stream_to_file(resp, dest_path, first_chunk=b""):
    """把响应流式写入文件（带进度）；完成后校验 Content-Length。"""
    tmp_path = dest_path + ".part"
    total = int(resp.headers.get("Content-Length") or 0)
    written = len(first_chunk)
    with open(tmp_path, "wb") as f:
        if first_chunk:
            f.write(first_chunk)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if total > 0:
                percent = min(100.0, written * 100.0 / total)
                print(
                    f"\r下载进度: {percent:5.1f}% ({written / 1e6:.1f} MB)",
                    end="",
                )
    print()
    if total > 0 and written != total:
        raise DownloadError(
            f"下载不完整：已写入 {written / 1e6:.1f} MB，"
            f"应为 {total / 1e6:.1f} MB。\n.part 已保留，请重跑本脚本。"
        )
    os.replace(tmp_path, dest_path)
    print(f"下载完成: {dest_path}")


def drive_download(file_id, dest_path):
    """走 Google Drive 大文件确认页流程（gdown 同款思路，纯标准库）。

    首次请求返回 HTML 确认页（含 name="confirm"），从中取 token 后
    改走 drive.usercontent.google.com 流式下载。
    """
    opener = build_opener()
    headers = {"User-Agent": USER_AGENT}

    first_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = open_url(urllib.request.Request(first_url, headers=headers), opener=opener)
    head = resp.read(4 * 1024 * 1024)

    if b"name=\"confirm\"" in head or b"uc-download-link" in head:
        m = re.search(rb'name="confirm" value="([0-9A-Za-z_-]+)"', head)
        confirm = m.group(1).decode() if m else "t"
        dl_url = (
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm={confirm}"
        )
        resp.close()
        resp = open_url(
            urllib.request.Request(dl_url, headers=headers), opener=opener
        )
        _stream_to_file(resp, dest_path)
    elif head[:2] == b"PK":
        # 小文件直接返回 zip 流
        _stream_to_file(resp, dest_path, first_chunk=head)
    else:
        resp.close()
        raise DownloadError(
            "Google Drive 返回了意外内容（链接失效或需要登录），"
            f"请检查 file id: {file_id}"
        )


def download_jhu_crowd(data_dir="data", keep_zip=False):
    """下载并解压 JHU-Crowd++ 到 data/jhu_crowd_v2.0。"""
    print(
        "JHU-Crowd++: 4,372 张图像（train 2,272 / val 500 / test 1,600），"
        "约 151 万标注，单图最多 25,791 人；压缩包约 2.87 GB"
    )
    os.makedirs(data_dir, exist_ok=True)

    target = os.path.join(data_dir, EXPECT_DIR)
    if os.path.isdir(target):
        print(f"{target} 已存在，跳过下载。")
        return

    zip_path = os.path.join(data_dir, ZIP_NAME)
    if not os.path.exists(zip_path):
        drive_download(DRIVE_FILE_ID, zip_path)
    else:
        print(f"已存在 {zip_path}，跳过下载。")

    extract_zip(zip_path, data_dir, expect_tops=(EXPECT_DIR,))

    if not keep_zip and os.path.exists(zip_path):
        os.remove(zip_path)
    print("JHU-Crowd++ 数据集准备完成。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载 JHU-Crowd++ 人群计数数据集（Google Drive 镜像）"
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
    download_jhu_crowd(**vars(parse_args()))
