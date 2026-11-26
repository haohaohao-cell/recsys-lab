"""导出精简数据包到 assets/demo/，供云端部署使用。

产物（约 9 MB，随仓库提交）：
    interactions.parquet  大矩阵正反馈（user_id, item_id, timestamp）
    dense.parquet         小矩阵正反馈，用于无偏评估
    items.parquet         视频元数据（caption + 三级类目）

原始数据（412 MB）不入库，看板在没有 raw 时自动回退到这份精简包。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEMO_DIR, WATCH_RATIO_THRESHOLD
from src.data_utils import _kuairec_dir, load_kuairec, load_kuairec_dense, load_kuairec_items


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    inter, items = load_kuairec()
    dense = load_kuairec_dense()

    # 只保留元数据里出现过的列，压缩后体积更小
    items = items[["item_id", "title", "genres"]].copy()
    items["title"] = items["title"].astype(str)
    items["genres"] = items["genres"].astype(str)

    for name, df in [("interactions", inter), ("dense", dense), ("items", items)]:
        out = DEMO_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False, compression="zstd")
        mb = out.stat().st_size / 1e6
        print(f"  {out.name:<24s} {len(df):>9,} 行  {mb:6.2f} MB")

    total = sum(f.stat().st_size for f in DEMO_DIR.glob("*.parquet")) / 1e6
    print(f"\n合计 {total:.1f} MB -> {DEMO_DIR}")
    print(f"正反馈阈值: watch_ratio > {WATCH_RATIO_THRESHOLD}")


if __name__ == "__main__":
    main()
