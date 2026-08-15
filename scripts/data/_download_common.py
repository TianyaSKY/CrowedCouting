"""scripts/data 下载脚本的公共工具（仅用标准库）。"""

import os
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
import zipfile

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


class DownloadError(RuntimeError):
    """下载/解压失败，消息面向终端用户。"""


def ssl_context():
    """优先使用 certifi CA 包（conda/部分发行版的 Python 不带系统 CA）。"""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def build_opener():
    """带 cookie jar 的 opener（SSL 走 certifi）。"""
    cj = urllib.request.HTTPCookieProcessor()
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context()),
        cj,
    )


def open_url(req, opener=None, timeout=None):
    """打开请求；证书校验失败时降级为不校验并告警（仅影响传输安全）。"""
    def _open():
        if opener is not None:
            return opener.open(req, timeout=timeout)
        return urllib.request.urlopen(req, context=ssl_context(), timeout=timeout)

    try:
        return _open()
    except (ssl.SSLError, urllib.error.URLError) as e:
        reason = getattr(e, "reason", e)
        if not isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" not in str(reason):
            raise
        print("警告: 服务器证书链不完整/系统 CA 缺失，本次跳过 SSL 校验（不影响数据内容）")
        if opener is not None:
            noverify = urllib.request.build_opener(
                urllib.request.HTTPSHandler(
                    context=ssl._create_unverified_context()
                )
            )
            return noverify.open(req)
        return urllib.request.urlopen(
            req, context=ssl._create_unverified_context()
        )


def download_file(url, dest_path, headers=None, opener=None, note=""):
    """带进度输出的单文件下载（支持断点续传 + 完整性校验）。

    - ``headers``: 附加请求头（默认带浏览器 UA）。
    - ``opener``: 传入带 cookie 的 opener 时使用流式下载。
    - ``note``: 附加在“保存到”后的说明（如文件大小）。
    - 已有 ``<dest>.part`` 时从该偏移发 Range 请求续传；下载完成后校验
      Content-Length，不完整则保留 .part 报错，重跑脚本即可续传。
    """
    print(f"开始下载: {url}")
    print(f"保存到: {dest_path}" + (f"（{note}）" if note else ""))

    tmp_path = dest_path + ".part"
    resume_from = (
        os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    )
    if resume_from > 0:
        print(f"发现未完成的 .part（{resume_from / 1e6:.1f} MB），从断点续传。")

    req_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    if resume_from > 0:
        req_headers["Range"] = f"bytes={resume_from}-"

    def report(downloaded, total_size):
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r下载进度: {percent:5.1f}% ({downloaded / 1e6:.1f} MB)", end="")

    try:
        req = urllib.request.Request(url, headers=req_headers)
        # idle timeout：服务器停发数据（连接挂死）时主动失败并保留 .part，
        # 避免无限挂起；正常下载（>=30s/1MB）不会触发。
        resp = open_url(req, opener=opener, timeout=30.0)
        status = getattr(resp, "status", 200)
        remaining = int(resp.headers.get("Content-Length") or 0)

        if status == 206:
            written = resume_from
            total_size = remaining + resume_from
            mode = "ab"
        else:
            written = 0
            total_size = remaining
            mode = "wb"

        with resp, open(tmp_path, mode) as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                report(written, total_size)
    except Exception as e:
        raise DownloadError(
            f"下载失败: {e}\n.part 已保留，可直接重跑本脚本续传。"
        ) from e

    if total_size > 0 and written != total_size:
        raise DownloadError(
            f"下载不完整：已写入 {written / 1e6:.1f} MB，"
            f"应为 {total_size / 1e6:.1f} MB。\n"
            f".part 已保留，可直接重跑本脚本续传。"
        )

    print()
    os.replace(tmp_path, dest_path)
    print(f"下载完成: {dest_path}")
    return dest_path


def extract_zip(zip_path, dest_dir, expect_tops=()):
    """解压 zip 并打印顶层目录；``expect_tops`` 用于校验镜像内容。"""
    print(f"解压 {os.path.basename(zip_path)} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        top_levels = {m.split("/", 1)[0] for m in members if "/" in m}
        print(f"zip 顶层目录: {sorted(top_levels)}")

        if expect_tops and not any(t in top_levels for t in expect_tops):
            raise DownloadError(
                f"zip 中未找到 {expect_tops}，请检查镜像源内容是否完整"
            )
        zf.extractall(dest_dir)


# (可执行名, 解压到当前目录的方式)。macOS 自带 bsdtar（即 tar）。
_RAR_TOOLS = [
    ("bsdtar", ["bsdtar", "-xf"]),
    ("tar", ["tar", "-xf"]),
    ("unar", ["unar", "-f"]),
    ("unrar", ["unrar", "x"]),
    ("7z", ["7z", "x"]),
]


def extract_rar(rar_path, dest_dir):
    """解压 RAR 归档（标准库不支持 rar，需借助系统工具）。"""
    print(f"解压 {os.path.basename(rar_path)} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    errors = []
    for name, cmd in _RAR_TOOLS:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(
                [*cmd, os.path.abspath(rar_path)],
                cwd=dest_dir, check=True, capture_output=True,
            )
            print(f"使用 {name} 解压完成。")
            return
        except subprocess.CalledProcessError as e:
            errors.append(
                f"{name}: {e.stderr.decode('utf-8', 'ignore').strip()[-200:]}"
            )

    detail = "；".join(errors) if errors else "未找到 bsdtar/tar/unar/unrar/7z"
    raise DownloadError(
        "RAR 解压失败。\n" + detail + "\n"
        "请安装任一工具后重试（macOS: brew install unar；Ubuntu: sudo apt install unrar）。"
    )
