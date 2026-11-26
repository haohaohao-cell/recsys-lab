"""流行度基线：按训练集交互次数排序。

作用不是「争第一」，而是给出一个下限参照：
  任何个性化模型在 HR@10 上如果打不过 Popular，说明实现有 bug 或数据有问题。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender


class Popular(Recommender):
    name = "Popular"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "Popular":
        self.n_users = n_users
        self.n_items = n_items
        counts = np.zeros(n_items, dtype=np.float32)
        np.add.at(counts, train["i"].to_numpy(), 1.0)
        # 同分排序的稳定性由 metrics.topk_indices 的稳定排序保证，这里不需要额外扰动
        self.counts = counts
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.counts, (len(users), 1))
