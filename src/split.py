"""时序切分与随机切分。

为什么默认用时序切分：
    随机切分会把「未来交互」放进训练集，评估时模型见过答案的邻域，
    指标会虚高一大截，且不同模型虚高幅度不同，导致模型排名失真。
    这是推荐系统复现研究里最常被诟病的错误（Dacrema et al., RecSys 2019）。
    本模块保留随机切分，只是为了让「协议差异」本身成为可量化的实验。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MIN_INTERACTIONS


@dataclass
class Split:
    train: pd.DataFrame        # 训练交互 [u, i, timestamp]
    val: pd.DataFrame          # 验证集（每用户 1 条）
    test: pd.DataFrame         # 测试集（每用户 1 条）
    n_users: int
    n_items: int
    user2idx: dict
    item2idx: dict

    def user_gt(self) -> dict[int, int]:
        return dict(zip(self.test["u"].to_numpy(), self.test["i"].to_numpy()))


def _finalize(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Split:
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    def _remap(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        out["u"] = out["user_id"].map(user2idx)
        out["i"] = out["item_id"].map(item2idx)
        return out[["u", "i", "timestamp"]].reset_index(drop=True)

    return Split(
        train=_remap(train),
        val=_remap(val),
        test=_remap(test),
        n_users=len(users),
        n_items=len(items),
        user2idx=user2idx,
        item2idx=item2idx,
    )


def leave_last_out(interactions: pd.DataFrame, min_interactions: int = MIN_INTERACTIONS) -> Split:
    """严格时序：每用户最后 1 条为 test，倒数第 2 条为 val，其余为 train。"""
    df = interactions.sort_values(["user_id", "timestamp"], kind="mergesort")
    counts = df.groupby("user_id")["item_id"].transform("size")
    df = df[counts >= min_interactions]

    rank = df.groupby("user_id").cumcount(ascending=False)
    return _finalize(df, df[rank >= 2], df[rank == 1], df[rank == 0])


def random_holdout(interactions: pd.DataFrame, min_interactions: int = MIN_INTERACTIONS, seed: int = 42) -> Split:
    """随机留一（会泄漏未来信息，仅用于量化「协议差异」）。"""
    rng = np.random.default_rng(seed)
    df = interactions.copy()
    counts = df.groupby("user_id")["item_id"].transform("size")
    df = df[counts >= min_interactions]
    df = df.assign(_r=rng.random(len(df))).sort_values(["user_id", "_r"], kind="mergesort")

    rank = df.groupby("user_id").cumcount(ascending=False)
    return _finalize(df, df[rank >= 2], df[rank == 1], df[rank == 0])
