"""ItemCF：基于物品共现的余弦相似度协同过滤。

实现要点（这些细节决定了它是不是一个「公平」的基线）：
  1. 只在训练集上共现，绝不碰 val/test；
  2. 相似度矩阵截断到 top-K 邻居，避免热门物品的噪声边主导打分；
  3. 对角线置零，防止「自己推自己」把已交互物品顶上来（虽然后面会 mask，
     但置零能让 K 个名额真正留给别的物品）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Recommender


class ItemCF(Recommender):
    def __init__(self, k_neighbors: int = 200, alpha: float = 0.5):
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.name = f"ItemCF(k={k_neighbors})"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "ItemCF":
        self.n_users, self.n_items = n_users, n_items

        u = train["u"].to_numpy()
        i = train["i"].to_numpy()
        R = sparse.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(n_users, n_items))
        R.data[:] = 1.0  # 二值化（隐式反馈）

        # 余弦相似度：C_ij / (|i|^alpha * |j|^alpha)，alpha=0.5 即标准余弦
        counts = np.asarray(R.sum(axis=0)).ravel().astype(np.float32)
        norm = np.zeros_like(counts)
        nz = counts > 0
        norm[nz] = 1.0 / np.power(counts[nz], self.alpha)

        Rn = R @ sparse.diags(norm)
        S = (Rn.T @ Rn).toarray().astype(np.float32)
        np.fill_diagonal(S, 0.0)

        # 截断到 top-K 邻居，并直接转成稀疏矩阵：
        # 10k 物品的稠密矩阵要 450MB，稀疏只有 ~25MB，且打分更快
        k = min(self.k_neighbors, n_items - 1)
        if k > 0:
            part = np.argpartition(S, -k, axis=1)[:, -k:]
            rows = np.repeat(np.arange(n_items), k)
            cols = part.ravel()
            vals = np.take_along_axis(S, part, axis=1).ravel()
            keep = vals > 0
            S = sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items), dtype=np.float32)
        else:
            S = sparse.csr_matrix(S)

        self.S = S
        self.R = R
        self.item_deg = counts
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        # 用户历史物品的相似度向量求和：score_u = sum_{i in history(u)} S[i]
        out = self.R[users] @ self.S
        if sparse.issparse(out):
            out = out.toarray()
        return np.asarray(out, dtype=np.float32)
