import argparse
import os
import shutil
import urllib.request
import zipfile

# 官方仓库 desenzhou/ShanghaiTechDataset 提供的下载源（2024 更新版）
DEFAULT_URL = (
    "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/"
    "ShanghaiTech_Crowd_Counting_Dataset.zip"
    "?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
)

# 备用源（需手动获取）：
# - 百度网盘: https://pan.baidu.com/s/1xJnhmJbwPdnNKBM1K6F1Cg?pwd=iga3
# - Kaggle:   https://www.kaggle.com/datasets/xyyu18/shanghaitech-crowd-counting-dataset


def download_file(url, dest_path):
    """带进度输出的单文件下载（仅用标准库）。"""
    print(f"开始下载: {url}")
    print(f"保存到: {dest_path}")

    tmp_path = dest_path + ".part"

    def report(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r下载进度: {percent:5.1f}% ({downloaded / 1e6:.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(
            url, tmp_path, reporthook=report
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"下载失败: {e}\n"
            "请检查网络，或改用 --url 指定其他镜像源。"
        )

    print()
    os.replace(tmp_path, dest_path)
    print(f"下载完成: {dest_path}")


def extract_zip(zip_path, dest_dir):
    """解压 zip，期望顶层包含 part_A_final/ 与 part_B_final/。"""
    print(f"解压 {zip_path} -> {dest_dir}")

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        top_levels = {
            m.split("/", 1)[0] for m in members if "/" in m
        }
        print(f"zip 顶层目录: {sorted(top_levels)}")

        if not any(
            t in {"part_A_final", "part_B_final"}
            for t in top_levels
        ):
            raise RuntimeError(
                "zip 中未找到 part_A_final/part_B_final，请检查镜像源内容"
            )

        zf.extractall(dest_dir)

    # 清理 zip 临时文件
    os.remove(zip_path)


def download_shanghaitech(url=None, data_dir="data", keep_zip=False):
    """下载并解压 ShanghaiTech 数据集到 data/part_A_final 与 data/part_B_final。"""
    url = url or DEFAULT_URL
    os.makedirs(data_dir, exist_ok=True)

    # 已存在则跳过
    if all(
        os.path.isdir(os.path.join(data_dir, name))
        for name in ("part_A_final", "part_B_final")
    ):
        print("data/part_A_final 与 data/part_B_final 已存在，跳过下载。")
        return

    zip_path = os.path.join(data_dir, "ShanghaiTech_Crowd_Counting_Dataset.zip")

    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    else:
        print(f"已存在 {zip_path}，跳过下载。")

    extract_zip(zip_path, data_dir)

    if not keep_zip and os.path.exists(zip_path):
        os.remove(zip_path)

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
