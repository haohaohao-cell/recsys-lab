"""D11 · 多路召回融合（召回层评估）

评估口径：
    召回层的产出是「候选集」，不是最终 10 条。所以公平的比较方式是
    **固定候选预算 B**，看 GT 物品能不能进入候选集：
        · 单路：SASRec top-B
        · 单路：Semantic top-B
        · 融合：SASRec top-B1 ∪ Semantic top-B2，且 B1 + B2 = B
    如果融合能在同等预算下捞到更多 GT，才算真的有效。

路由策略：
    语义召回整体很弱，但长尾桶非零。所以额外测一个「只让语义召回负责长尾」的版本：
    先取 SASRec top-B，再用语义召回补 B2 个**训练集中出现次数 <= T** 的长尾候选。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import topk_indices
from src.models.semantic import Semantic
from src.split import leave_last_out
from scripts.semantic_recall import build_texts, encode_tfidf


def main() -> None:
    inter, items = load_dataset()
    split = leave_last_out(inter)
    n_items = split.n_items

    test_gt = split.user_gt()
    seen: dict[int, set[int]] = {}
    both = pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])
    for u, i in zip(both["u"].to_numpy(), both["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))

    pop = split.train.groupby("i").size().reindex(range(n_items), fill_value=0).to_numpy()

    # ---------------- 模型 ----------------
    import torch

    from src.models.sasrec import SASRec, _Net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = SASRec(hidden=64, max_len=50, device=device)
    seq.net = _Net(n_items, 64, 50, 2, 2, 0.2).to(device)
    seq.net.load_state_dict(torch.load(RESULTS / "sasrec.pt", map_location=device))
    seq.net.eval()
    hist = pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]])
    seq.set_history(hist, split.n_users, n_items)
    seq.name = "SASRec"

    texts = build_texts(items, split, "category")
    V = encode_tfidf(texts, 256)
    sem = Semantic(V, mode="knn", k=100, name="Semantic").fit(split.train, split.n_users, n_items)

    users = np.array(sorted(test_gt.keys()))
    B = 130
    TAIL_MAX = 20  # 训练集中出现次数 <= 20 视为长尾

    def topk_of(scores: np.ndarray, u: int, k: int) -> list[int]:
        s = scores.copy()
        s[list(seen.get(u, set()))] = -np.inf
        return topk_indices(s[None, :], k)[0].tolist()

    stats = {name: [] for name in [
        "SASRec@130（全部给序列）",
        "SASRec@100 + 语义30",
        "SASRec@100 + 语义长尾30（路由）",
        "语义@130",
    ]}

    sasrec_cache: dict[int, list[int]] = {}
    sem_cache: dict[int, list[int]] = {}
    batch = 512

    for s in range(0, len(users), batch):
        ub = users[s : s + batch]
        sc_s = np.asarray(seq.score(ub), dtype=np.float32)
        sc_m = np.asarray(sem.score(ub), dtype=np.float32)
        for r, u in enumerate(ub):
            u = int(u)
            sasrec_cache[u] = topk_of(sc_s[r], u, B)
            sem_cache[u] = topk_of(sc_m[r], u, B)

    for u in users:
        u = int(u)
        gt = test_gt[u]
        s_list = sasrec_cache[u]
        m_list = sem_cache[u]
        tail_from_sem = [i for i in m_list if pop[i] <= TAIL_MAX][:30]

        cands = {
            "SASRec@130（全部给序列）": set(s_list),
            "SASRec@100 + 语义30": set(s_list[:100]) | set(m_list[:30]),
            "SASRec@100 + 语义长尾30（路由）": set(s_list[:100]) | set(tail_from_sem),
            "语义@130": set(m_list),
        }
        for name, cs in cands.items():
            stats[name].append(int(gt in cs))

    print("=" * 84)
    print(f"召回层评估（候选预算固定 130，{len(users):,} 个测试用户）")
    print("=" * 84)
    print(f"{'策略':<30s}{'整体命中率':>12s}{'长尾 GT':>11s}{'中部 GT':>11s}{'头部 GT':>11s}")

    gts = np.array([test_gt[int(u)] for u in users])
    p = pop[gts]
    tail_cut = np.quantile(p, 0.2)
    head_cut = np.quantile(p, 0.6)
    mask_tail, mask_head = p <= tail_cut, p > head_cut
    mask_mid = ~mask_tail & ~mask_head

    for name in stats:
        hit = np.array(stats[name], dtype=bool)
        print(f"{name:<30s}{hit.mean():>12.4f}{hit[mask_tail].mean():>11.4f}"
              f"{hit[mask_mid].mean():>11.4f}{hit[mask_head].mean():>11.4f}")

    base = np.array(stats["SASRec@130（全部给序列）"], dtype=bool)
    print("\n相对「130 全给序列模型」的增量：")
    for name in stats:
        if name.startswith("SASRec@130"):
            continue
        h = np.array(stats[name], dtype=bool)
        print(f"  {name:<28s} {(h.sum() - base.sum()) / len(users) * 100:+.3f} 个百分点"
              f"   （新增 {int((h & ~base).sum())}，丢失 {int((base & ~h).sum())}）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {name: float(np.mean(stats[name])) for name in stats}
    with open(RESULTS / "fusion_recall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.Series(out).to_csv(RESULTS / "metrics_fusion.csv")
    print(f"\n已保存 {RESULTS / 'fusion_recall.json'}")


if __name__ == "__main__":
    main()
