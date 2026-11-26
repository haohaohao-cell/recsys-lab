"""在验证集上调超参，再报测试集 —— 超参不碰 test。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_RAW
from src.data_utils import load_ml1m
from src.metrics import evaluate, summarize
from src.models.itemcf import ItemCF
from src.split import leave_last_out


def seen_from(df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for u, i in zip(df["u"].to_numpy(), df["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


inter, _ = load_ml1m(DATA_RAW)
split = leave_last_out(inter)

# 验证集：候选屏蔽 train 中的物品
val_gt = dict(zip(split.val["u"].to_numpy(), split.val["i"].to_numpy()))
val_seen = seen_from(split.train)
test_gt = split.user_gt()
test_seen = seen_from(pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]]))

rows = []
for k in (20, 50, 100, 200, 500):
    for alpha in (0.3, 0.5, 0.7):
        m = ItemCF(k_neighbors=k, alpha=alpha).fit(split.train, split.n_users, split.n_items)
        s = summarize(evaluate(m.score, split.n_users, val_gt, val_seen, split.n_items, (10,), 256), (10,))
        rows.append({"k": k, "alpha": alpha, **s})

val_df = pd.DataFrame(rows).sort_values("HR@10", ascending=False)
print("验证集超参搜索（按 HR@10）")
print(val_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best = val_df.iloc[0]
print(f"\n最优: k={int(best['k'])}, alpha={best['alpha']}  (val HR@10={best['HR@10']:.4f})")

m = ItemCF(k_neighbors=int(best["k"]), alpha=float(best["alpha"])).fit(split.train, split.n_users, split.n_items)
s = summarize(evaluate(m.score, split.n_users, test_gt, test_seen, split.n_items, (10, 20), 256), (10, 20))
print(f"对应测试集: HR@10={s['HR@10']:.4f}  NDCG@10={s['NDCG@10']:.4f}  HR@20={s['HR@20']:.4f}")
