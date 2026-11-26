"""KuaiRec 数据探查：规模、稀疏度、正样本比例、稠密子集覆盖度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import WATCH_RATIO_THRESHOLD

BASE = r"D:\reclab-data\raw\kuairec\KuaiRec 2.0\data"
USE = ["user_id", "video_id", "watch_ratio", "timestamp"]
DT = {"user_id": np.int32, "video_id": np.int32, "watch_ratio": np.float32, "timestamp": np.float64}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{BASE}\\{name}.csv", usecols=USE, dtype=DT)


def report(df: pd.DataFrame, name: str) -> None:
    nu, ni = df["user_id"].nunique(), df["video_id"].nunique()
    print(f"\n=== {name}")
    print(f"  交互 {len(df):,} | 用户 {nu:,} | 视频 {ni:,} | 覆盖率 {len(df) / (nu * ni):.2%}")
    print(f"  watch_ratio 分位: " + " ".join(
        f"p{q}={df['watch_ratio'].quantile(q / 100):.2f}" for q in (10, 50, 75, 90, 99)
    ))
    pos = (df["watch_ratio"] > WATCH_RATIO_THRESHOLD).sum()
    print(f"  watch_ratio > {WATCH_RATIO_THRESHOLD}: {pos:,} ({pos / len(df):.2%})")
    per_user = df.groupby("user_id").size()
    print(f"  人均交互 {per_user.mean():.1f} | 中位 {per_user.median():.0f} | "
          f"最少 {per_user.min()} | 最多 {per_user.max()}")
    ts = pd.to_datetime(df["timestamp"], unit="s")
    print(f"  时间范围 {ts.min()} ~ {ts.max()}")


big = load("big_matrix")
report(big, "big_matrix（常规曝光，有偏）")

small = load("small_matrix")
report(small, "small_matrix（稠密曝光，接近全量）")

# 稠密子集的真实覆盖度：1411 用户 × 3327 视频 的官方口径
dense_users = small["user_id"].nunique()
dense_items = small["video_id"].nunique()
print(f"\n稠密子集矩阵: {dense_users} 用户 × {dense_items} 视频 = {dense_users * dense_items:,} 格, "
      f"实际观测 {len(small):,} 格 -> 覆盖 {len(small) / (dense_users * dense_items):.2%}")

# 两个矩阵的用户/视频重叠
print(f"big 独有用户 {len(set(big['user_id']) - set(small['user_id'])):,}, "
      f"small 独有用户 {len(set(small['user_id']) - set(big['user_id'])):,}")
print(f"big 独有视频 {len(set(big['video_id']) - set(small['video_id'])):,}, "
      f"small 独有视频 {len(set(small['video_id']) - set(big['video_id'])):,}")
