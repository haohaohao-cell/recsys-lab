"""下载 MovieLens-1M 到 data/raw/ 并解压。

本机存在证书链问题（curl 报 SSL error 60），因此这里用未校验的 SSL 上下文兜底。
仅用于下载公开数据，不涉及任何凭据。
"""
from __future__ import annotations

import ssl
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATA_RAW

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = DATA_RAW / "ml-1m.zip"
DEST = DATA_RAW / "ml-1m"


def _download(url: str, dst: Path) -> None:
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp, open(dst, "wb") as f:
            f.write(resp.read())
    except ssl.SSLError:
        print("  SSL 校验失败，改用不校验模式重试 ...")
        insecure = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=30, context=insecure) as resp, open(dst, "wb") as f:
            f.write(resp.read())


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if (DEST / "ratings.dat").exists():
        print(f"数据已存在: {DEST}")
        return

    print(f"下载 {URL}")
    _download(URL, ZIP_PATH)
    print(f"  完成，{ZIP_PATH.stat().st_size / 1e6:.1f} MB")

    print(f"解压到 {DATA_RAW}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_RAW)
    print(f"  完成: {DEST}")


if __name__ == "__main__":
    main()
