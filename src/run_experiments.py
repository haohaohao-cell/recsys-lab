"""实验入口：加载 -> 时序切分 -> 训练基线 -> 全量排序评估 -> 分组分析。

用法：
    python -m src.run_experiments
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import EVAL_BATCH, KS, RESULTS
from .data_utils import describe, load_dataset
from .metrics import covered_items, evaluate, group_by_item_popularity, group_by_user_activity, summarize
from .models.itemcf import ItemCF
from .models.popular import Popular
from .split import leave_last_out


def build_gt(split) -> tuple[dict[int, int], dict[int, set[int]]]:
    gt = dict(zip(split.test["u"].to_numpy(), split.test["i"].to_numpy()))
    seen: dict[int, set[int]] = {}
    both = pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])
    for u, i in zip(both["u"].to_numpy(), both["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return gt, seen


def run() -> pd.DataFrame:
    print("=" * 68)
    print("加载数据 ...")
    inter, items = load_dataset()
    stats = describe(inter)
    print(
        f"  交互 {stats['n_interactions']:,} | 用户 {stats['n_users']:,} | "
        f"物品 {stats['n_items']:,} | 稀疏度 {stats['density']:.5%} | 人均 {stats['avg_len']:.1f}"
    )

    split = leave_last_out(inter)
    print(
        f"  切分后: train {len(split.train):,} | val {len(split.val):,} | "
        f"test {len(split.test):,} | 用户 {split.n_users:,} | 物品 {split.n_items:,}"
    )

    gt, seen = build_gt(split)

    models = [
        Popular(),
        ItemCF(k_neighbors=200),
        ItemCF(k_neighbors=500, alpha=0.7),  # 验证集调参所得（见 scripts/tune_itemcf.py）
    ]
    rows, details = [], {}

    for m in models:
        t0 = time.time()
        m.fit(split.train, split.n_users, split.n_items)
        fit_s = time.time() - t0

        t0 = time.time()
        df = evaluate(m.score, split.n_users, gt, seen, split.n_items, KS, EVAL_BATCH)
        eval_s = time.time() - t0

        s = summarize(df, KS)
        s["model"] = m.name
        s["coverage@10"] = covered_items(m.score, split.n_users, split.n_items, seen, 10)
        s["fit_s"] = round(fit_s, 2)
        s["eval_s"] = round(eval_s, 2)
        rows.append(s)

        df = group_by_user_activity(df, split.train)
        df = group_by_item_popularity(df, split.train, gt, split.test)
        details[m.name] = df
        print(f"\n[{m.name}] HR@10={s['HR@10']:.4f}  NDCG@10={s['NDCG@10']:.4f}  "
              f"耗时 fit={fit_s:.1f}s eval={eval_s:.1f}s")

    res = pd.DataFrame(rows).set_index("model")
    cols = [c for c in ["HR@10", "NDCG@10", "HR@20", "NDCG@20", "MRR", "coverage@10", "fit_s", "eval_s"] if c in res]
    res = res[cols]

    print("\n" + "=" * 68)
    print("总体指标（全量排序 / leave-last-out）")
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- 分组分析：这才是有信息量的部分 ----
    print("\n" + "-" * 68)
    print("按用户活跃度分组（HR@10）")
    act = pd.DataFrame({name: d.groupby("activity", observed=True)["hr@10"].mean() for name, d in details.items()})
    print(act.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n按被预测物品的流行度分组（HR@10）")
    tail = pd.DataFrame({name: d.groupby("item_bucket", observed=True)["hr@10"].mean() for name, d in details.items()})
    tail = tail.reindex(["长尾", "中部", "头部"])
    print(tail.to_string(float_format=lambda x: f"{x:.4f}"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "metrics_overall.csv")
    for name, d in details.items():
        safe = name.replace("(", "_").replace(")", "").replace("=", "")
        d.to_csv(RESULTS / f"per_user_{safe}.csv", index=False)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(res.round(4).to_dict(orient="index"), f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 {RESULTS}")
    return res


if __name__ == "__main__":
    run()
