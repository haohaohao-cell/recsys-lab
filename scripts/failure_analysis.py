"""错例分析：四个策略对每个测试用户，把「真实下一条」排到了第几名。

产物：results/failure_ranks.csv（每个用户一行）
    user, rank_SASRec, rank_ItemCF, rank_Popular, rank_Semantic,
    best_rank, n_hit, gt_pop, hist_len

用途：
    · 找出「全员失败」的用户（四个模型都没进 top-10）
    · 量化失败归因：是物品太冷门，还是用户历史太短
    · 区分「差一点」和「完全够不到」——前者说明排序可优化，后者说明信号不存在
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.split import leave_last_out
from src.text import build_texts

STRATS = ["SASRec", "ItemCF", "Popular", "Semantic"]
BATCH = 512


def main() -> None:
    inter, items = load_dataset()
    split = leave_last_out(inter)
    n_items = split.n_items

    gt = split.user_gt()
    users = np.array(sorted(gt.keys()))
    gt_items = np.array([gt[int(u)] for u in users])

    hidden: dict[int, list[int]] = {}
    for u, i in zip(*[pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])[c].to_numpy() for c in ("u", "i")]):
        hidden.setdefault(int(u), []).append(int(i))

    # ---------------- 模型 ----------------
    from src.models.itemcf import ItemCF
    from src.models.popular import Popular

    models: dict[str, object] = {}
    models["Popular"] = Popular().fit(split.train, split.n_users, n_items)
    models["ItemCF"] = ItemCF(k_neighbors=200).fit(split.train, split.n_users, n_items)

    from src.models.semantic import Semantic

    texts = build_texts(items, split, "category")
    models["Semantic"] = _semantic(texts, split)

    import torch

    from src.models.sasrec import SASRec, _Net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = SASRec(hidden=64, max_len=50, device=device)
    seq.net = _Net(n_items, 64, 50, 2, 2, 0.2).to(device)
    seq.net.load_state_dict(torch.load(RESULTS / "sasrec.pt", map_location=device))
    seq.net.eval()
    seq.set_history(
        pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]]),
        split.n_users, n_items,
    )
    models["SASRec"] = seq

    # ---------------- 排名 ----------------
    ranks = {name: np.zeros(len(users), dtype=np.int64) for name in STRATS}

    for s in range(0, len(users), BATCH):
        ub = users[s : s + BATCH]
        for name, model in models.items():
            sc = np.asarray(model.score(ub), dtype=np.float32)
            for r, u in enumerate(ub):
                h = hidden.get(int(u))
                if h:
                    sc[r, h] = -np.inf
            own = sc[np.arange(len(ub)), gt_items[s : s + BATCH]]
            ranks[name][s : s + BATCH] = (sc > own[:, None]).sum(axis=1) + 1
        print(f"  {min(s + BATCH, len(users)):,}/{len(users):,}", end="\r")

    pop = split.train.groupby("i").size().reindex(range(n_items), fill_value=0).to_numpy()
    hist_len = np.array([len(hidden.get(int(u), [])) for u in users])

    df = pd.DataFrame({"user": users, **{f"rank_{k}": v for k, v in ranks.items()}})
    df["best_rank"] = df[[f"rank_{k}" for k in STRATS]].min(axis=1)
    df["n_hit"] = (df[[f"rank_{k}" for k in STRATS]] <= 10).sum(axis=1)
    df["gt_pop"] = pop[gt_items]
    df["hist_len"] = hist_len
    df["all_fail"] = df["n_hit"] == 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "failure_ranks.csv", index=False)

    n = len(df)
    print("\n" + "=" * 66)
    print("命中分布（四个策略，各取 top-10）")
    print("=" * 66)
    for k in range(5):
        c = int((df["n_hit"] == k).sum())
        print(f"  {k} 个模型命中：{c:6,}  ({c / n:6.2%})")

    print("\n" + "=" * 66)
    print("全员失败的用户，失败在哪")
    print("=" * 66)
    fail = df[df["all_fail"]]
    ok = df[~df["all_fail"]]
    print(f"  全员失败 {len(fail):,} 人（{len(fail) / n:.1%}）")
    print(f"  其中「差一点」（最佳排名 11-50）：{((fail['best_rank'] > 10) & (fail['best_rank'] <= 50)).mean():.1%}")
    print(f"  其中「完全够不到」（最佳排名 > 1000）：{(fail['best_rank'] > 1000).mean():.1%}")
    print()
    print(f"  真实下一条的流行度（训练集中出现次数）")
    print(f"    失败组 中位数 {fail['gt_pop'].median():6.0f}  均值 {fail['gt_pop'].mean():7.1f}")
    print(f"    命中组 中位数 {ok['gt_pop'].median():6.0f}  均值 {ok['gt_pop'].mean():7.1f}")
    print(f"  失败组里 GT 流行度 <= 5 的占比：{(fail['gt_pop'] <= 5).mean():.1%}")
    print(f"  命中组里 GT 流行度 <= 5 的占比：{(ok['gt_pop'] <= 5).mean():.1%}")
    print()
    print(f"  用户历史长度  失败组中位 {fail['hist_len'].median():.0f} | 命中组中位 {ok['hist_len'].median():.0f}")

    print(f"\n已保存 {RESULTS / 'failure_ranks.csv'}")


def _semantic(texts, split):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from src.models.semantic import Semantic

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, max_features=30000).fit_transform(texts)
    Z = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=42).fit_transform(X)
    V = normalize(Z).astype(np.float32)
    return Semantic(V, mode="knn", k=100).fit(split.train, split.n_users, split.n_items)


if __name__ == "__main__":
    main()
