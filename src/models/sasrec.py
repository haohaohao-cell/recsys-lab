"""SASRec：基于自注意力（Self-Attentive）的序列推荐。

与原始论文的差异（有意为之，都会在报告里说明）：
  1. 论文用「1 个正样本 + 1 个采样负样本」做 BCE；这里直接对**全词表做交叉熵**。
     原因是我们的评估本身就是全量排序，采样负例会引入与评估不一致的偏差。
     代价是每个 step 的显存更高，所以 batch 取小一些。
  2. 输入/输出 embedding 共享权重（相当于语言模型的 weight tying）。
  3. 用 pre-LN 残差结构，训练更稳定，不需要论文里的 warmup 也能收敛。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Recommender


class _Block(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class _Net(nn.Module):
    def __init__(self, n_items: int, hidden: int, max_len: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        # index 0 保留给 padding
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_Block(hidden, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.item_emb.weight[0])
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) 左填充的 item 序列 -> logits (B, L, n_items+1)。"""
        b, length = ids.shape
        pos = torch.arange(length, device=ids.device).unsqueeze(0)
        x = self.drop(self.item_emb(ids) + self.pos_emb(pos))

        # 因果掩码用有限大负数：用 -inf 时，全被 mask 的行会让 softmax 出 NaN
        attn_mask = torch.triu(
            torch.full((length, length), -1e9, device=ids.device, dtype=x.dtype), diagonal=1
        )
        # padding 掩码也转成加性浮点掩码，与 attn_mask 类型一致（避免 torch 的 deprecation 警告）
        pad_mask = (ids == 0).to(x.dtype) * -1e9
        for blk in self.blocks:
            x = blk(x, attn_mask, pad_mask)
        x = self.ln(x)
        return x @ self.item_emb.weight.t()


class SASRec(Recommender):
    def __init__(
        self,
        hidden: int = 64,
        max_len: int = 50,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 128,
        patience: int = 8,
        device: str | None = None,
        seed: int = 42,
    ):
        self.hidden, self.max_len = hidden, max_len
        self.n_layers, self.n_heads, self.dropout = n_layers, n_heads, dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience, self.seed = patience, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"SASRec(h{hidden},L{max_len})"
        self.val_fn = None  # 由外部注入：fn(epoch) -> float，用于早停

    # ---------------- 历史序列 ----------------
    def _history_from(self, df: pd.DataFrame, n_users: int, n_items: int) -> list[np.ndarray]:
        self.n_users, self.n_items = n_users, n_items
        df = df.sort_values(["u", "timestamp"], kind="mergesort")
        hist: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
        for u, g in df.groupby("u", sort=False):
            hist[int(u)] = g["i"].to_numpy(dtype=np.int64)[-self.max_len :]
        return hist

    def _pad_batch(self, users: np.ndarray) -> torch.Tensor:
        """把 users 的历史序列左填充成 (B, max_len) 张量。"""
        ids = np.zeros((len(users), self.max_len), dtype=np.int64)
        for r, u in enumerate(users):
            h = self.hist[int(u)][-self.max_len :]
            if len(h):
                ids[r, self.max_len - len(h) :] = h + 1  # +1 是因为 0 留给 padding
        return torch.from_numpy(ids).to(self.device)

    # ---------------- 训练 ----------------
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "SASRec":
        self.hist = self._history_from(train, n_users, n_items)
        self.n_items = n_items

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        ids = self._pad_batch(np.arange(n_users))
        inp, tgt = ids[:, :-1], ids[:, 1:]
        valid = tgt != 0

        # 排除「左侧完全没有真实物品」的位置：这些位置的注意力被全 mask，梯度是噪声
        first_real = torch.where((ids != 0).any(dim=1), (ids != 0).float().argmax(dim=1), torch.zeros(n_users, dtype=torch.long, device=self.device))
        pos = torch.arange(inp.size(1), device=self.device).unsqueeze(0)
        valid = valid & (pos >= first_real.unsqueeze(1))

        self.net = _Net(n_items, self.hidden, self.max_len, self.n_layers, self.n_heads, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

        n = n_users
        best, best_state, bad = -1.0, None, 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            total, count = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                x, y, m = inp[idx], tgt[idx], valid[idx]
                if not m.any():
                    continue
                logits = self.net(x)
                loss = F.cross_entropy(logits[m], y[m])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += loss.item() * int(m.sum())
                count += int(m.sum())

            msg = f"  epoch {epoch:3d}  loss={total / max(count, 1):.4f}"
            if self.val_fn is not None:
                score = self.val_fn(epoch)
                msg += f"  val HR@10={score:.4f}"
                sched.step(score)
                if score > best:
                    best, bad = score, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= self.patience:
                        print(msg + "  -> 早停")
                        break
            print(msg)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    # ---------------- 推理 ----------------
    def set_history(self, df: pd.DataFrame, n_users: int, n_items: int) -> None:
        """切换打分所用历史（验证时用 train，测试时用 train+val）。"""
        self.hist = self._history_from(df, n_users, n_items)

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        self.net.eval()
        ids = self._pad_batch(np.asarray(users))
        logits = self.net(ids)          # (B, L, V+1)
        last = logits[:, -1, :]         # 最后一个位置代表「下一个物品」
        return last[:, 1:].float().cpu().numpy()  # 去掉 padding 列
