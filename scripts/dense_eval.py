"""D9 · 稠密矩阵上的多正样本 Recall@K

为什么需要这个评估：
    之前的 next-item 评测每个用户只有 1 个正样本，Recall@K 退化成 HitRate@K，
    衡量的是「能不能猜中紧接着的那一条」。
    稠密矩阵给了每个用户约 154 个真实喜欢的视频，于是可以算真正的 Recall@K：
    推荐的 K 条里，覆盖了他 154 个兴趣中的几个。

数据口径：
    测试集 = 1,411 个稠密用户 × 3,327 个稠密视频
    正样本 = 该用户在这些视频里 watch_ratio > 2 的那些（人均约 154 个）
    候选   = 全部 3,327 个（稠密视频按构造就与用户的常规历史不重叠，无需再 mask）
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset, load_kuairec_dense
from src.metrics import topk_indices
from src.models.itemcf import ItemCF
from src.models.popular import Popular
from src.split import leave_last_out

KS = (10, 20, 50)


def main() -> None:
    inter, items = load_dataset()
    split = leave_last_out(inter)
    dense = load_kuairec_dense()

    # 视频 id -> 模型内部列号；稠密视频里可能有训练集没出现过的，需要剔除
    dense_vid = np.sort(dense["item_id"].unique())
    cols, kept_vid = [], []
    for v in dense_vid:
        if int(v) in split.item2idx:
            kept_vid.append(int(v))
            cols.append(split.item2idx[int(v)])
    cols = np.array(cols)
    n_dropped = len(dense_vid) - len(cols)

    # 用户 id -> 内部行号
    dense_uid = np.sort(dense["user_id"].unique())
    users = np.array([u for u in dense_uid if int(u) in split.user2idx])
    print(f"稠密矩阵: {len(dense):,} 条 | 用户 {len(dense_uid):,} | 视频 {len(dense_vid):,}")
    print(f"可评估: 用户 {len(users):,} | 视频 {len(cols):,}"
          f"（{n_dropped} 个视频在训练集中未出现，已剔除）")
    print(f"人均正样本 {len(dense) / len(dense_uid):.1f} 个\n")

    # 每个用户的真值集合（用稠密视频 id 表示）
    gt: dict[int, set[int]] = {}
    for u, v in zip(dense["user_id"].to_numpy(), dense["item_id"].to_numpy()):
        if int(u) in split.user2idx:
            gt.setdefault(int(u), set()).add(int(v))

    # ---------------- 模型 ----------------
    models = [
        Popular().fit(split.train, split.n_users, split.n_items),
        ItemCF(k_neighbors=200).fit(split.train, split.n_users, split.n_items),
    ]

    try:
        import torch

        from src.models.sasrec import SASRec, _Net

        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq = SASRec(hidden=64, max_len=50, device=device)
        seq.net = _Net(split.n_items, 64, 50, 2, 2, 0.2).to(device)
        seq.net.load_state_dict(torch.load(RESULTS / "sasrec.pt", map_location=device))
        seq.net.eval()
        hist = pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]])
        seq.set_history(hist, split.n_users, split.n_items)
        seq.name = "SASRec"
        models.append(seq)
    except Exception as e:
        print(f"（SASRec 加载失败，跳过：{e}）")

    rows = []
    uidx = np.array([split.user2idx[int(u)] for u in users])
    for m in models:
        scores = np.asarray(m.score(uidx), dtype=np.float32)[:, cols]  # (n_users, n_dense_items)
        hit_counts = {k: 0 for k in KS}
        recalls = {k: [] for k in KS}

        for r, u in enumerate(users):
            top = topk_indices(scores[r : r + 1], max(KS))[0]
            top_vids = [kept_vid[c] for c in top]
            truth = gt.get(int(u), set())
            for k in KS:
                hit = len(set(top_vids[:k]) & truth)
                recalls[k].append(hit / max(len(truth), 1))
                hit_counts[k] += hit

        row = {"模型": m.name if hasattr(m, "name") else type(m).__name__}
        for k in KS:
            row[f"Recall@{k}"] = float(np.mean(recalls[k]))
        row["平均命中数@20"] = hit_counts[20] / len(users)
        rows.append(row)
        print(f"[{row['模型']}] " + "  ".join(f"Recall@{k}={row[f'Recall@{k}']:.4f}" for k in KS))

    res = pd.DataFrame(rows)

    # 随机基线：从 3,327 个候选里随机取 K 个，期望召回率 = K / n_items
    n_items_dense = len(cols)
    rand = {k: k / n_items_dense for k in KS}
    res.loc[len(res)] = {"模型": "Random(理论)", **{f"Recall@{k}": rand[k] for k in KS},
                         "平均命中数@20": 20 * len(dense) / len(dense_uid) / n_items_dense}

    res = res.set_index("模型")
    print("\n" + "=" * 70)
    print(f"多正样本 Recall@K（{len(users):,} 用户 × {n_items_dense:,} 视频，人均 "
          f"{len(dense) / len(dense_uid):.0f} 个真值）")
    print("=" * 70)
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))

    # 相对随机的倍数
    print("\n相对随机基线的倍数：")
    for k in KS:
        base = res.loc["Random(理论)", f"Recall@{k}"]
        for m in res.index:
            if m == "Random(理论)":
                continue
            print(f"  {m:<12s} Recall@{k}: {res.loc[m, f'Recall@{k}'] / base:6.2f}×")

    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "dense_eval.csv")
    with open(RESULTS / "dense_eval.json", "w", encoding="utf-8") as f:
        json.dump(res.round(4).to_dict(orient="index"), f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {RESULTS / 'dense_eval.csv'}")


if __name__ == "__main__":
    main()
