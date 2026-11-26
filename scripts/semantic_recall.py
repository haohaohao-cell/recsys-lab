"""语义召回：用视频文案 + 类目做内容召回，并在两种协议下评估。

用法：
    python -m scripts.semantic_recall --encoder tfidf   # TF-IDF + SVD（无需下载）
    python -m scripts.semantic_recall --encoder bge     # BGE 中文向量（需下载模型）
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset, load_kuairec_dense
from src.metrics import evaluate, group_by_item_popularity, summarize, topk_indices
from src.models.semantic import Semantic
from src.split import leave_last_out
from src.text import build_texts


def encode_tfidf(texts: list[str], dim: int = 256) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, max_features=30000)
    X = vec.fit_transform(texts)
    n_comp = min(dim, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)
    print(f"  TF-IDF 词表 {X.shape[1]:,} 维 -> SVD {n_comp} 维，"
          f"解释方差 {svd.explained_variance_ratio_.sum():.1%}")
    return normalize(Z).astype(np.float32)


def encode_bge(texts: list[str]) -> np.ndarray:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer

    name = "BAAI/bge-small-zh-v1.5"
    print(f"  加载 {name} ...")
    m = SentenceTransformer(name)
    Z = m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    print(f"  BGE 编码完成: {Z.shape}")
    return np.asarray(Z, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["tfidf", "bge"], default="tfidf")
    ap.add_argument("--source", choices=["caption", "category", "both"], default="both")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--mode", choices=["knn", "mean"], default="knn")
    ap.add_argument("--k", type=int, default=100, help="knn 模式下每个物品保留的邻居数")
    ap.add_argument("--decay", type=float, default=0.0, help="历史衰减强度，0 表示等权")
    args = ap.parse_args()

    inter, items = load_dataset()
    split = leave_last_out(inter)
    print(f"数据: {len(inter):,} 交互 | {split.n_users:,} 用户 | {split.n_items:,} 视频")

    texts = build_texts(items, split, args.source)
    empty = sum(1 for t in texts if not t.strip())
    print(f"文本: 共 {len(texts):,} 条，空文本 {empty} 条")

    print(f"编码方式: {args.encoder}")
    V = encode_tfidf(texts, args.dim) if args.encoder == "tfidf" else encode_bge(texts)

    model = Semantic(V, mode=args.mode, k=args.k, decay=args.decay,
                     name=f"Semantic-{args.encoder.upper()}-{args.mode}")

    # ---------------- 协议一：next-item ----------------
    model.fit(split.train, split.n_users, split.n_items)
    test_gt = split.user_gt()
    test_seen: dict[int, set[int]] = {}
    both = pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])
    for u, i in zip(both["u"].to_numpy(), both["i"].to_numpy()):
        test_seen.setdefault(int(u), set()).add(int(i))

    df = evaluate(model.score, split.n_users, test_gt, test_seen, split.n_items, (10, 20), 256)
    s = summarize(df, (10, 20))
    print(f"\n[next-item] HR@10={s['HR@10']:.4f}  NDCG@10={s['NDCG@10']:.4f}")

    # 长尾分组：协同过滤在长尾桶是 0，语义召回只要能拿到非 0 就有价值
    dfb = group_by_item_popularity(df, split.train, test_gt, split.test)
    bucket = (
        dfb.groupby("item_bucket", observed=True)["hr@10"].mean()
        .reindex(["长尾", "中部", "头部"]).fillna(0.0)
    )
    print("[长尾分组] " + "  ".join(f"{k}={v:.4f}" for k, v in bucket.items()))

    # ---------------- 协议二：稠密多正样本 ----------------
    dense = load_kuairec_dense()
    dense_vid = np.sort(dense["item_id"].unique())
    cols = np.array([split.item2idx[int(v)] for v in dense_vid if int(v) in split.item2idx])
    kept = [int(v) for v in dense_vid if int(v) in split.item2idx]

    users = np.array([u for u in np.sort(dense["user_id"].unique()) if int(u) in split.user2idx])
    gt: dict[int, set[int]] = {}
    for u, v in zip(dense["user_id"].to_numpy(), dense["item_id"].to_numpy()):
        if int(u) in split.user2idx:
            gt.setdefault(int(u), set()).add(int(v))

    uidx = np.array([split.user2idx[int(u)] for u in users])
    scores = np.asarray(model.score(uidx), dtype=np.float32)[:, cols]

    recalls = {10: [], 20: [], 50: []}
    for r, u in enumerate(users):
        top = topk_indices(scores[r : r + 1], 50)[0]
        top_vids = [kept[c] for c in top]
        truth = gt.get(int(u), set())
        for k in recalls:
            recalls[k].append(len(set(top_vids[:k]) & truth) / max(len(truth), 1))

    print(f"[dense]     Recall@10={np.mean(recalls[10]):.4f}  "
          f"Recall@20={np.mean(recalls[20]):.4f}  Recall@50={np.mean(recalls[50]):.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "encoder": args.encoder,
        "dim": int(V.shape[1]),
        "decay": args.decay,
        "next_item": {k: round(float(v), 4) for k, v in s.items()},
        "dense": {f"Recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()},
    }
    tag = f"semantic_{args.encoder}"
    df.to_csv(RESULTS / f"per_user_semantic_{args.source}.csv", index=False)
    with open(RESULTS / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.Series({**out["next_item"], **out["dense"]}).to_csv(RESULTS / f"metrics_{tag}.csv")
    print(f"\n已保存 {RESULTS / f'{tag}.json'}")


if __name__ == "__main__":
    main()
