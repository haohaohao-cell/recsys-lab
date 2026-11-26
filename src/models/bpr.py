"""BPR-MF：贝叶斯个性化排序的矩阵分解（Rendle et al., UAI 2009）。

这是深度学习推荐最经典的对照基线：
  若一个 Transformer 序列模型打不过 BPR，通常说明数据里的协同信号本身很弱，
  或者评估协议有问题 —— 这两种情况都必须能解释清楚。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import Recommender


class BPR(Recommender):
    def __init__(
        self,
        dim: int = 64,
        epochs: int = 50,
        batch_size: int = 4096,
        lr: float = 1e-3,
        reg: float = 1e-5,
        patience: int = 8,
        device: str | None = None,
        seed: int = 42,
    ):
        self.dim, self.epochs, self.batch_size = dim, epochs, batch_size
        self.lr, self.reg, self.patience = lr, reg, patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.name = f"BPR-MF(d{dim})"
        self.val_fn = None

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "BPR":
        self.n_users, self.n_items = n_users, n_items
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        u = torch.from_numpy(train["u"].to_numpy().astype(np.int64)).to(self.device)
        i = torch.from_numpy(train["i"].to_numpy().astype(np.int64)).to(self.device)
        n = len(u)

        self.user_emb = nn.Embedding(n_users, self.dim).to(self.device)
        self.item_emb = nn.Embedding(n_items, self.dim).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        opt = torch.optim.Adam(
            list(self.user_emb.parameters()) + list(self.item_emb.parameters()), lr=self.lr
        )

        best, best_state, bad = -1.0, None, 0

        for epoch in range(1, self.epochs + 1):
            perm = torch.from_numpy(rng.permutation(n)).to(self.device)
            total, steps = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                bu, bi = u[idx], i[idx]
                bj = torch.randint(0, n_items, (len(idx),), device=self.device)

                eu = self.user_emb(bu)
                ei = self.item_emb(bi)
                ej = self.item_emb(bj)

                x = (eu * (ei - ej)).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(x).mean()
                reg = (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / len(idx)

                opt.zero_grad(set_to_none=True)
                (loss + self.reg * reg).backward()
                opt.step()

                total += loss.item()
                steps += 1

            msg = f"  epoch {epoch:3d}  loss={total / max(steps, 1):.4f}"
            if self.val_fn is not None:
                score = self.val_fn(epoch)
                msg += f"  val HR@10={score:.4f}"
                if score > best:
                    best, bad = score, 0
                    best_state = {
                        "u": copy.deepcopy(self.user_emb.weight.detach()),
                        "i": copy.deepcopy(self.item_emb.weight.detach()),
                    }
                else:
                    bad += 1
                    if bad >= self.patience:
                        print(msg + "  -> 早停")
                        break
            print(msg)

        if best_state is not None:
            with torch.no_grad():
                self.user_emb.weight.copy_(best_state["u"])
                self.item_emb.weight.copy_(best_state["i"])
        return self

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        bu = torch.from_numpy(np.asarray(users).astype(np.int64)).to(self.device)
        return (self.user_emb(bu) @ self.item_emb.weight.t()).float().cpu().numpy()
