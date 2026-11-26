"""下载 KuaiRec 数据集到 data/raw/kuairec/ 并解压。

用法：
    python -m scripts.download_kuairec

约 412 MB，支持断点续传（网络中断后重跑即可）。
"""
from __future__ import annotations

import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_RAW

URL = "https://zenodo.org/api/records/18164998/files/KuaiRec.zip/content"
EXPECTED = 431_964_858  # 字节数，用于校验完整性
ZIP_PATH = DATA_RAW / "KuaiRec.zip"
DEST = DATA_RAW / "kuairec"
SENTINEL = DEST / "KuaiRec 2.0" / "data" / "big_matrix.csv"


def _ctx() -> ssl.SSLContext:
    return ssl.create_default_context()


def _download(url: str, dst: Path) -> None:
    """带断点续传的下载。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    if have >= EXPECTED:
        print(f"  已下载完整（{have / 1e6:.0f} MB），跳过")
        return

    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0, last = time.time(), 0.0
    try:
        resp = urllib.request.urlopen(req, timeout=60, context=_ctx())
    except urllib.error.URLError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        print("  SSL 校验失败，改用不校验模式重试 ...")
        resp = urllib.request.urlopen(req, timeout=60, context=ssl._create_unverified_context())

    mode = "ab" if have and resp.status == 206 else "wb"
    total = have + int(resp.headers.get("Content-Length") or 0)
    print(f"  下载中：{have / 1e6:.0f} MB 已有，目标 {total / 1e6:.0f} MB")

    with open(dst, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            if time.time() - last > 2:
                speed = have / 1e6 / max(time.time() - t0, 1)
                pct = have / EXPECTED * 100
                print(f"\r  {have / 1e6:7.1f} MB  ({pct:5.1f}%)  {speed:5.2f} MB/s", end="")
                last = time.time()
    print(f"\r  {have / 1e6:7.1f} MB  (100.0%) 完成          ")


def main() -> None:
    if SENTINEL.exists():
        print(f"数据已存在：{DEST}")
        return

    print("KuaiRec 数据集（快手短视频，约 412 MB）")
    print("来源：https://zenodo.org/records/18164998\n")

    _download(URL, ZIP_PATH)

    size = ZIP_PATH.stat().st_size
    if size < EXPECTED:
        print(f"⚠ 文件不完整（{size:,} / {EXPECTED:,} 字节），请重跑本脚本续传。")
        sys.exit(1)

    print("解压 ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DEST)

    if not SENTINEL.exists():
        print("⚠ 解压完成但未找到 big_matrix.csv，请检查目录结构。")
        sys.exit(1)

    print(f"完成：{DEST}")
    print("下一步：python -m scripts.train_sasrec && streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
