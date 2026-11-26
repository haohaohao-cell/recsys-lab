"""协议敏感性实验：同一份数据、同一个模型，只换切分方式，指标差多少？

这是本项目最有说服力的一个结论：
  「ML-1M 上 SASRec HR@10 能到 0.7」这类数字，大多来自随机切分 + 负采样评估，
  其中训练集包含测试物品的时间邻域，属于信息泄漏。换成严格时序切分后，
  同一个 ItemCF 的 HR@10 只有 0.06 量级。
  结论不是「模型变差了」，而是「评测协议变了」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import evaluate, summarize
from src.models.itemcf import ItemCF
from src.models.popular import Popular
from src.split import leave_last_out, random_holdout


def seen_from(df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for u, i in zip(df["u"].to_numpy(), df["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


inter, _ = load_dataset()
rows = []

for name, split in [
    ("随机留一（泄漏未来）", random_holdout(inter, seed=42)),
    ("时序留最后（严格）", leave_last_out(inter)),
]:
    gt = split.user_gt()
    seen = seen_from(pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]]))
    for m in (Popular(), ItemCF(k_neighbors=200, alpha=0.5)):
        m.fit(split.train, split.n_users, split.n_items)
        s = summarize(evaluate(m.score, split.n_users, gt, seen, split.n_items, (10, 20), 256), (10, 20))
        rows.append({"协议": name, "模型": m.name, **s})

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

piv = df.pivot(index="模型", columns="协议", values="HR@10")
piv["倍差"] = piv["随机留一（泄漏未来）"] / piv["时序留最后（严格）"]
print("\nHR@10 协议差异倍数：")
print(piv.to_string(float_format=lambda x: f"{x:.2f}"))

from src.config import RESULTS

RESULTS.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS / "protocol_compare.csv", index=False)
print(f"\n已保存 {RESULTS / 'protocol_compare.csv'}")
