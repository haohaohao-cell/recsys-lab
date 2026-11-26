"""演示：拿真实用户，看各路召回到底在推荐什么。

用法：
    python -m scripts.show_demo          # 挑一个 SASRec 猜中的用户（便于讲解）
    python -m scripts.show_demo --any    # 随机口径，更接近真实命中率
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import topk_indices
from src.models.itemcf import ItemCF
from src.models.popular import Popular
from src.split import leave_last_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--any", action="store_true")
    args = ap.parse_args()

    inter, items = load_dataset()
    split = leave_last_out(inter)

    idx2item = {v: k for k, v in split.item2idx.items()}
    title = dict(zip(items["item_id"], items["title"]))

    def name(i: int) -> str:
        return str(title.get(idx2item[int(i)], f"item-{int(i)}"))[:44]

    gt = split.user_gt()
    lens = split.train.groupby("u").size()
    pop = split.train.groupby("i").size()

    # 候选：历史长度适中、下一部不是无人问津的冷门视频
    cand = [u for u in gt if 60 <= lens.get(u, 0) <= 200 and 30 <= pop.get(gt[u], 0) <= 800]
    cand = sorted(cand)[:500]

    import torch

    from src.models.sasrec import SASRec, _Net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = SASRec(hidden=64, max_len=50, device=device)
    seq.net = _Net(split.n_items, 64, 50, 2, 2, 0.2).to(device)
    seq.net.load_state_dict(torch.load(RESULTS / "sasrec.pt", map_location=device))
    seq.net.eval()
    hist_all = pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]])
    seq.set_history(hist_all, split.n_users, split.n_items)

    seen_map: dict[int, set[int]] = {}
    for uu, ii in zip(hist_all["u"].to_numpy(), hist_all["i"].to_numpy()):
        seen_map.setdefault(int(uu), set()).add(int(ii))

    users = np.array(cand)
    scores = seq.score(users)
    hits = []
    for r, uu in enumerate(users):
        sc = scores[r].copy()
        sc[list(seen_map.get(int(uu), set()))] = -np.inf
        top = topk_indices(sc[None, :], 10)[0]
        pos = np.where(top == gt[int(uu)])[0]
        hits.append(int(pos[0]) if len(pos) else -1)
    hits = np.array(hits)

    print(f"这批 {len(cand)} 个用户的 SASRec 命中率：top-10 内 {np.mean(hits >= 0):.1%}，"
          f"top-3 内 {np.mean((hits >= 0) & (hits <= 2)):.1%}")

    if args.any:
        u = int(cand[len(cand) // 2])
    else:
        good = [int(u) for u, h in zip(cand, hits) if 0 <= h <= 2]
        u = good[len(good) // 2] if good else int(cand[len(cand) // 2])
    print(f"（展示样本命中排名 = {hits[cand.index(u)] + 1 if hits[cand.index(u)] >= 0 else '未命中'}）")

    hist = split.train[split.train["u"] == u].sort_values("timestamp")
    print("\n" + "=" * 78)
    print(f"用户 #{u}  有效观看 {len(hist)} 条，最近 8 条：")
    for t in hist["i"].to_numpy()[-8:]:
        print(f"    · {name(t)}")
    print(f"\n他接下来真的看了：{name(gt[u])}")
    print("=" * 78)

    seen = seen_map.get(u, set())

    pop_model = Popular().fit(split.train, split.n_users, split.n_items)
    sp = pop_model.score(np.array([u]))[0].copy()
    sp[list(seen)] = -np.inf
    top_pop = topk_indices(sp[None, :], 5)[0]

    itemcf = ItemCF(k_neighbors=200).fit(split.train, split.n_users, split.n_items)
    sc_cf = itemcf.score(np.array([u]))[0].copy()
    sc_cf[list(seen)] = -np.inf
    top_cf = topk_indices(sc_cf[None, :], 5)[0]

    sc = seq.score(np.array([u]))[0].copy()
    sc[list(seen)] = -np.inf
    top_seq = topk_indices(sc[None, :], 5)[0]

    rows = [("Popular", top_pop), ("ItemCF", top_cf), ("SASRec", top_seq)]
    print()
    for rank in range(5):
        line = f"#{rank + 1}  "
        for label, arr in rows:
            mark = " ★" if arr[rank] == gt[u] else "  "
            line += f"{name(arr[rank]):<46s}{mark} "
        print(line)
    print(" " * 5 + "  ".join(f"{label:<48s}" for label, _ in rows))


if __name__ == "__main__":
    main()
