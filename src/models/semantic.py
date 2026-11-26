"""语义召回：用物品的文本表示做内容召回。

两种打分方式：
    mean  用户画像 = 历史物品向量的平均
          问题：118 个不同视频平均出来的向量接近「平均口味」，和什么都不太像。
    knn   内容版 ItemCF：先算物品-物品的内容相似度，再对用户历史求和
          更符合「这个视频和他看过的某一个像不像」的召回语义。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Recommender


class Semantic(Recommender):
    """item_vectors: (n_items, d)，行对应内部物品下标。"""

    def __init__(
        self,
        item_vectors: np.ndarray,
        mode: str = "knn",
        k: int = 100,
        decay: float = 0.0,
        name: str = "Semantic",
    ):
        V = np.asarray(item_vectors, dtype=np.float32)
        n = np.linalg.norm(V, axis=1, keepdims=True)
        n[n == 0] = 1.0
        self.V = V / n
        self.mode, self.k, self.decay = mode, k, decay
        self.name = name

    @property
    def dim(self) -> int:
        return self.V.shape[1]

    # ---------------------------------------------------------------- 相似度
    def _build_sim(self, block: int = 1024) -> sparse.csr_matrix:
        """逐块计算 (n_items, n_items) 的 top-k 内容相似度矩阵，避免 450MB 的稠密矩阵。"""
        n = self.V.shape[0]
        k = min(self.k, n - 1)
        rows, cols, vals = [], [], []

        for s in range(0, n, block):
            sim = self.V[s : s + block] @ self.V.T  # (b, n)
            # 对角线置零，避免「自己推自己」
            for r in range(sim.shape[0]):
                sim[r, s + r] = 0.0
            top = np.argpartition(-sim, k, axis=1)[:, :k]
            topv = np.take_along_axis(sim, top, axis=1)
            rows.append(np.repeat(np.arange(s, s + sim.shape[0]), k))
            cols.append(top.ravel())
            vals.append(topv.ravel())

        data = np.concatenate(vals)
        row_idx = np.concatenate(rows)
        col_idx = np.concatenate(cols)
        keep = data > 0  # 丢弃负相似度，避免把不相关的物品也加进来
        return sparse.csr_matrix(
            (data[keep], (row_idx[keep], col_idx[keep])), shape=(n, n), dtype=np.float32
        )

    # ---------------------------------------------------------------- 训练
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "Semantic":
        self.n_users, self.n_items = n_users, n_items
        u = train["u"].to_numpy()
        i = train["i"].to_numpy()

        if self.decay > 0:
            df = pd.DataFrame({"u": u, "i": i}).sort_values(["u"])
            pos = df.groupby("u").cumcount()
            length = df.groupby("u")["u"].transform("size")
            w = np.exp(-self.decay * (length - 1 - pos).to_numpy()).astype(np.float32)
        else:
            w = np.ones(len(u), dtype=np.float32)

        self.R = sparse.csr_matrix((w, (u, i)), shape=(n_users, n_items), dtype=np.float32)

        if self.mode == "mean":
            U = np.asarray(self.R @ self.V, dtype=np.float32)
            norm = np.linalg.norm(U, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            self.U = U / norm
        else:
            self.S = self._build_sim()
        return self

    # ---------------------------------------------------------------- 打分
    def score(self, users: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            return self.U[users] @ self.V.T
        return np.asarray((self.R[users] @ self.S).todense(), dtype=np.float32)
