"""构建 dashboard 所需的全部产物（一次加载，全局复用）。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .config import DATA_PROC, RESULTS
from .data_utils import load_dataset
from .metrics import topk_indices
from .models.itemcf import ItemCF
from .models.popular import Popular
from .split import leave_last_out
from .text import build_texts, clean_text

CACHE_DIR = DATA_PROC / "app_cache"


def _cached(name: str, builder):
    """磁盘缓存：首次构建后落盘，之后加载只要零点几秒。

    拟合一个 ItemCF/语义模型要十几秒，看板每次启动都重算会让人以为卡死了。
    """
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # 类定义变了就重建
    obj = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return obj


@dataclass
class Bundle:
    split: object
    title: dict[int, str]
    genres: dict[int, str]
    models: dict[str, object]
    pop: np.ndarray
    hist: dict[int, list[int]]
    seen: dict[int, set[int]]
    gt: dict[int, int]
    sim_csc: dict[str, sparse.csc_matrix] = field(default_factory=dict)
    dense_stats: dict = field(default_factory=dict)


def _genre_display(g: str) -> str:
    """类目展示：去掉 UNKNOWN 占位符，保留有效层级。"""
    parts = [p.strip() for p in str(g).split("|") if p.strip() and p.strip() != "UNKNOWN"]
    return " · ".join(parts) if parts else "未分类"


def _load_sasrec(n_items: int):
    import torch
    from .models.sasrec import SASRec, _Net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SASRec(hidden=64, max_len=50, device=device)
    m.net = _Net(n_items, 64, 50, 2, 2, 0.2).to(device)
    m.net.load_state_dict(torch.load(RESULTS / "sasrec.pt", map_location=device))
    m.net.eval()
    m.name = "SASRec"
    return m


def build_bundle() -> Bundle:
    inter, items = load_dataset()
    split = leave_last_out(inter)

    from .data_utils import load_kuairec_dense

    dense = load_kuairec_dense()
    n_du, n_di = dense["user_id"].nunique(), dense["item_id"].nunique()
    dense_stats = {
        "n_users": int(n_du),
        "n_items": int(n_di),
        "coverage": float(len(dense) / (n_du * n_di)),
        "pos_total": int(len(dense)),
        "pos_per_user": float(len(dense) / n_du),
    }

    idx2item = {v: k for k, v in split.item2idx.items()}
    raw_title = dict(zip(items["item_id"], items["title"]))
    raw_genre = dict(zip(items["item_id"], items["genres"]))
    title = {i: clean_text(raw_title.get(idx2item[i], "")) or "（无文案）" for i in range(split.n_items)}
    genres = {i: _genre_display(raw_genre.get(idx2item[i], "")) for i in range(split.n_items)}

    hist: dict[int, list[int]] = {}
    for u, g in split.train.sort_values(["u", "timestamp"], kind="mergesort").groupby("u", sort=False):
        hist[int(u)] = g["i"].to_numpy().tolist()

    seen: dict[int, set[int]] = {}
    both = pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])
    for u, i in zip(both["u"].to_numpy(), both["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))

    pop = split.train.groupby("i").size().reindex(range(split.n_items), fill_value=0).to_numpy()

    # ---------------- 模型 ----------------
    models: dict[str, object] = {}
    models["Popular"] = Popular().fit(split.train, split.n_users, split.n_items)

    itemcf = _cached("itemcf_k200", lambda: ItemCF(k_neighbors=200).fit(split.train, split.n_users, split.n_items))
    models["ItemCF"] = itemcf
    # 只看 top-k 邻居的稀疏副本，用于「因为你看过 X」的解释
    sim_csc: dict[str, sparse.csc_matrix] = {"ItemCF": sparse.csc_matrix(itemcf.S).tocsc()}

    from .models.semantic import Semantic

    sem = _cached(
        "semantic_category",
        lambda: Semantic(_encode_semantic(items, split), mode="knn", k=100, name="语义召回")
        .fit(split.train, split.n_users, split.n_items),
    )
    models["语义召回"] = sem
    sim_csc["语义召回"] = sem.S.tocsc()

    try:
        seq = _load_sasrec(split.n_items)
        hist_all = pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]])
        seq.set_history(hist_all, split.n_users, split.n_items)
        models["SASRec"] = seq
    except Exception as e:  # pragma: no cover
        print(f"SASRec 加载失败：{e}")

    return Bundle(
        split=split, title=title, genres=genres, models=models, pop=pop,
        hist=hist, seen=seen, gt=split.user_gt(), sim_csc=sim_csc, dense_stats=dense_stats,
    )


def _encode_semantic(items: pd.DataFrame, split) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = build_texts(items, split, "category")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, max_features=30000)
    X = vec.fit_transform(texts)
    Z = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=42).fit_transform(X)
    return normalize(Z).astype(np.float32)


# ---------------------------------------------------------------- 推荐
def recommend(b: Bundle, u: int, strategy: str, k: int = 10) -> list[dict]:
    model = b.models[strategy]
    scores = np.asarray(model.score(np.array([u])), dtype=np.float32)[0].copy()
    hidden = b.seen.get(int(u))
    if hidden:
        scores[list(hidden)] = -np.inf
    top = topk_indices(scores[None, :], k)[0]

    gt = b.gt.get(int(u))
    out = []
    for rank, i in enumerate(top, 1):
        out.append({
            "rank": rank,
            "item": int(i),
            "title": b.title[int(i)],
            "genre": b.genres[int(i)],
            "hit": int(i) == gt,
            "why": explain(b, strategy, int(u), int(i)),
        })
    return out


def explain(b: Bundle, strategy: str, u: int, item: int) -> str:
    if strategy == "Popular":
        return "全站最热"
    if strategy == "SASRec":
        n = min(len(b.hist.get(u, [])), 8)
        return f"顺着最近 {n} 条观看"
    csc = b.sim_csc.get(strategy)
    if csc is None:
        return ""
    col = csc[:, item]
    hist = set(b.hist.get(u, []))
    best, best_sim = None, -1.0
    for h, s in zip(col.indices, col.data):
        if int(h) in hist and s > best_sim:
            best, best_sim = int(h), float(s)
    if best is None:
        return "内容相近" if strategy == "语义召回" else "共现信号弱"
    return f"{'内容接近' if strategy == '语义召回' else '因为你看过'}「{b.title[best][:16]}」"


# ---------------------------------------------------------------- 诊断数据
def bucket_table(b: Bundle) -> pd.DataFrame:
    """读取各模型的逐用户结果，按物品流行度分桶。"""
    from .metrics import group_by_item_popularity

    pop_s = b.pop
    gt_item = pd.Series(b.gt)
    frames = {}
    for f in sorted(RESULTS.glob("per_user_*.csv")):
        key = f.stem.replace("per_user_", "")
        df = pd.read_csv(f)
        df["gt_item"] = df["user"].map(gt_item)
        df["gt_pop"] = df["gt_item"].map(lambda x: pop_s[int(x)] if pd.notna(x) else 0)
        q20, q60 = df["gt_pop"].quantile([0.2, 0.6])
        df["bucket"] = np.where(df["gt_pop"] <= q20, "长尾", np.where(df["gt_pop"] <= q60, "中部", "头部"))
        frames[key] = df.groupby("bucket")["hr@10"].mean()
    out = pd.DataFrame(frames).reindex(["长尾", "中部", "头部"])
    return out
