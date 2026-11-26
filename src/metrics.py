"""评估指标：全量排序下的 HR@K / NDCG@K / MRR / 覆盖率 / 长尾分组。

协议说明：
  * leave-one-out 下每个用户只有 1 个正样本，因此 Recall@K == HR@K，
    这里只报 HR@K，避免误导。
  * 候选集 = 全体物品 - 该用户在 train+val 中已交互的物品（不做负采样）。
    采样评估会系统性高估热门物品、低估长尾，与真实线上表现偏离。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / np.log2(ranks + 1.0)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """稳定取 top-k。

    用 argpartition 时，同分物品的先后顺序依赖实现细节，换 numpy 版本就可能变，
    指标会在小数点后 4 位漂移。这里改用稳定排序：同分时按下标升序，
    保证同一份数据在任何环境下得到完全相同的数字。
    """
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(
    score_fn,
    n_users: int,
    ground_truth: dict[int, int],
    seen: dict[int, set[int]],
    n_items: int,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 256,
) -> pd.DataFrame:
    """对每个用户做全量排序，返回逐用户指标明细。

    score_fn(user_idx: np.ndarray) -> np.ndarray, shape (len(user_idx), n_items)
    """
    rows = []
    users = np.array(sorted(ground_truth.keys()))

    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = score_fn(batch).astype(np.float32, copy=True)

        # 屏蔽已交互物品
        for r, u in enumerate(batch):
            hidden = seen.get(int(u))
            if hidden:
                scores[r, list(hidden)] = -np.inf

        # 取最大 K 的候选（稳定排序，同分按下标）
        k_max = max(ks)
        topk = topk_indices(scores, k_max)

        for r, u in enumerate(batch):
            gt = ground_truth[int(u)]
            hits = np.where(topk[r] == gt)[0]
            rank = int(hits[0]) + 1 if len(hits) else None  # 1-based

            rec = {"user": int(u), "rank": rank if rank is not None else 10**9}
            for k in ks:
                rec[f"hr@{k}"] = float(rank is not None and rank <= k)
                rec[f"ndcg@{k}"] = (
                    float(_dcg(np.array([rank]))[0]) if (rank is not None and rank <= k) else 0.0
                )
            rec["mrr"] = 1.0 / rank if rank is not None else 0.0
            rows.append(rec)

    return pd.DataFrame(rows)


def covered_items(score_fn, n_users: int, n_items: int, seen: dict, k: int = 10, batch_size: int = 256) -> int:
    """推荐列表覆盖的物品数（衡量是否千人一面）。"""
    seen_items: set[int] = set()
    users = np.arange(n_users)
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = score_fn(batch).astype(np.float32, copy=True)
        for r, u in enumerate(batch):
            hidden = seen.get(int(u))
            if hidden:
                scores[r, list(hidden)] = -np.inf
        topk = topk_indices(scores, k)
        seen_items.update(int(x) for x in np.unique(topk))
    return len(seen_items)


def summarize(df: pd.DataFrame, ks: tuple[int, ...] = (10, 20)) -> dict:
    out = {}
    for k in ks:
        out[f"HR@{k}"] = df[f"hr@{k}"].mean()
        out[f"NDCG@{k}"] = df[f"ndcg@{k}"].mean()
    out["MRR"] = df["mrr"].mean()
    return out


def group_by_user_activity(df: pd.DataFrame, train: pd.DataFrame, bins=(0, 20, 100, 10**9)) -> pd.DataFrame:
    """按训练集交互数分组，看模型对不同活跃度用户的效果差异。"""
    counts = train.groupby("u").size()
    df = df.copy()
    df["user_len"] = df["user"].map(counts).fillna(0).astype(int)
    labels = ["冷(<20)", "中(20-100)", "热(>100)"]
    df["activity"] = pd.cut(df["user_len"], bins=list(bins), labels=labels, right=False)
    return df


def group_by_item_popularity(df: pd.DataFrame, train: pd.DataFrame, ground_truth: dict[int, int], test: pd.DataFrame) -> pd.DataFrame:
    """按「被预测物品」的训练集流行度分桶，衡量模型对长尾的覆盖能力。"""
    pop = train.groupby("i").size()
    df = df.copy()
    gt_item = test.set_index("u")["i"]
    df["gt_item"] = df["user"].map(gt_item)
    df["gt_pop"] = df["gt_item"].map(pop).fillna(0).astype(int)

    # 按分位数分桶：头部 20% / 中部 40% / 长尾 40%
    q20, q60 = df["gt_pop"].quantile([0.2, 0.6])
    df["item_bucket"] = np.where(
        df["gt_pop"] <= q20, "长尾", np.where(df["gt_pop"] <= q60, "中部", "头部")
    )
    return df
