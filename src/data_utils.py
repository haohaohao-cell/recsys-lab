"""数据加载：MovieLens-1M 与 KuaiRec，统一成 (user_id, item_id, timestamp) + 物品文本。

两个数据集的差别只体现在本文件里，切分、评估、模型全部与数据集无关。

KuaiRec 的正反馈定义：
    watch_ratio = 播放时长 / 视频时长，> 2 表示用户看了两遍以上，是 KuaiRec 论文
    与多数后续工作采用的「有效观看」口径。它比「点开就算」严格得多——大矩阵里
    只有 7.5% 的曝光明细满足这个条件。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW, DEFAULT_DATASET, MIN_RATING, WATCH_RATIO_THRESHOLD


# ---------------------------------------------------------------- MovieLens
def load_ml1m(raw_dir: Path = DATA_RAW, min_rating: float = MIN_RATING) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml = Path(raw_dir) / "ml-1m"
    ratings = pd.read_csv(
        ml / "ratings.dat", sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"], encoding="latin-1",
    )
    movies = pd.read_csv(
        ml / "movies.dat", sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="latin-1",
    )
    inter = (
        ratings.loc[ratings["rating"] >= min_rating, ["user_id", "item_id", "timestamp"]]
        .drop_duplicates(subset=["user_id", "item_id"])
        .sort_values(["user_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    items = movies.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return inter, items


# ---------------------------------------------------------------- KuaiRec
def _kuairec_dir(raw_dir: Path) -> Path:
    base = Path(raw_dir) / "kuairec"
    for cand in base.glob("KuaiRec*/data"):
        return cand
    raise FileNotFoundError(f"未找到 KuaiRec 数据目录（{base}）")


def _read_matrix(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["user_id", "video_id", "watch_ratio", "timestamp"],
        dtype={"user_id": np.int32, "video_id": np.int32, "watch_ratio": np.float32, "timestamp": np.float64},
    )
    keep = df["watch_ratio"] > threshold
    out = df.loc[keep, ["user_id", "video_id", "timestamp"]].rename(columns={"video_id": "item_id"})
    return out.drop_duplicates(subset=["user_id", "item_id"]).sort_values(
        ["user_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def load_kuairec(
    raw_dir: Path = DATA_RAW,
    threshold: float = WATCH_RATIO_THRESHOLD,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """大矩阵（常规有偏曝光）-> 正反馈交互表 + 视频元数据。"""
    cache = Path(DATA_PROC) / f"kuairec_big_pos_{threshold}.parquet"
    if use_cache and cache.exists():
        inter = pd.read_parquet(cache)
    else:
        inter = _read_matrix(_kuairec_dir(raw_dir) / "big_matrix.csv", threshold)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            inter.to_parquet(cache, index=False)
    return inter, load_kuairec_items(raw_dir)


def load_kuairec_dense(
    raw_dir: Path = DATA_RAW,
    threshold: float = WATCH_RATIO_THRESHOLD,
    use_cache: bool = True,
) -> pd.DataFrame:
    """小矩阵（99.6% 全曝光）-> 用于「无偏评估」的真值反馈表。"""
    cache = Path(DATA_PROC) / f"kuairec_small_pos_{threshold}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    dense = _read_matrix(_kuairec_dir(raw_dir) / "small_matrix.csv", threshold)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        dense.to_parquet(cache, index=False)
    return dense


def load_kuairec_exposure(
    raw_dir: Path = DATA_RAW,
    users: np.ndarray | list[int] | None = None,
    use_cache: bool = True,
) -> dict[int, set[int]]:
    """大矩阵的**原始曝光记录**（不做 watch_ratio 过滤）。

    与 load_kuairec 的区别很关键：
      - load_kuairec        -> 「有效观看」= 喜欢，用于训练与评测
      - load_kuairec_exposure -> 「是否被推送过」= 曝光，用于识别假负例
    一个视频可能被曝光但用户立刻划走（watch_ratio 很低），它依然算「曝光过」。
    """
    cache = Path(DATA_PROC) / ("kuairec_big_exposure_all.parquet" if users is None else "kuairec_big_exposure_dense.parquet")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = pd.read_csv(
            _kuairec_dir(raw_dir) / "big_matrix.csv",
            usecols=["user_id", "video_id"],
            dtype={"user_id": np.int32, "video_id": np.int32},
        ).drop_duplicates()
        if users is not None:
            df = df[df["user_id"].isin(set(int(x) for x in users))]
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)

    out: dict[int, set[int]] = {}
    for u, v in zip(df["user_id"].to_numpy(), df["video_id"].to_numpy()):
        out.setdefault(int(u), set()).add(int(v))
    return out


def load_kuairec_items(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """视频元数据：caption（文案）+ 三级类目，合并成一个可用于文本编码的 title。"""
    cat = pd.read_csv(_kuairec_dir(raw_dir) / "kuairec_caption_category.csv")
    # 原始文件里有少量 video_id 缺失的脏行，先剔除
    vid = pd.to_numeric(cat["video_id"], errors="coerce")
    cat = cat.loc[vid.notna()].copy()
    cat["video_id"] = vid[vid.notna()].astype(np.int32)

    def s(col: str) -> pd.Series:
        return cat[col].fillna("").astype(str).str.strip()

    cap, l1, l2, l3 = s("caption"), s("first_level_category_name"), s("second_level_category_name"), s("third_level_category_name")

    # 优先用视频文案；文案缺失时退化为类目名，保证每个物品都有可编码的文本
    title = np.where(cap != "", cap, np.where(l2 != "", l2, np.where(l1 != "", l1, "未知")))
    genres = (l1 + "|" + l2 + "|" + l3).str.strip("|")

    items = pd.DataFrame(
        {"item_id": cat["video_id"].astype(np.int32), "title": title, "genres": genres}
    ).drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    return items


# ---------------------------------------------------------------- 统一入口
def load_dataset(name: str = DEFAULT_DATASET, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name == "ml1m":
        return load_ml1m(**kwargs)
    if name == "kuairec":
        return load_kuairec(**kwargs)
    raise ValueError(f"未知数据集: {name}")


def describe(interactions: pd.DataFrame) -> dict:
    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    n = len(interactions)
    return {
        "n_interactions": n,
        "n_users": n_users,
        "n_items": n_items,
        "density": n / (n_users * n_items),
        "avg_len": n / n_users,
        "median_len": interactions.groupby("user_id").size().median(),
    }
