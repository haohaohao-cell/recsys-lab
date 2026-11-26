"""D8 · 曝光偏差量化：算法推荐 vs 随机曝光

背景（这是 KuaiRec 最独特的地方）：
    对 1,411 个用户，快手额外做了一次「随机曝光」实验——把这 3,327 个视频
    直接推给他们，不管算法会不会推。于是同一批用户身上同时有两种数据：
        · 常规信息流：平台算法挑选的 ~390 条内容
        · 随机曝光：  随机分配的 3,327 条内容
    这构成一个天然的对照实验，可以回答一个平时只能靠 A/B 测试回答的问题：
        「推荐算法相对随机投放，到底提升了多少？」

指标：
    有效率 = P(watch_ratio > 阈值 | 被曝光)
    · 算法有效率   = 常规信息流里的有效率
    · 随机有效率   = 随机曝光里的有效率（相当于「不分青红皂白地推」）
    · 增益         = 算法有效率 / 随机有效率

    同时，随机有效率还有第二层含义：训练时我们把「没曝光」的物品当负样本，
    而随机抽样下其中有 4.6% 左右其实是正样本 —— 这就是负采样的标签噪声下限。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_PROC, DATA_RAW, RESULTS, WATCH_RATIO_THRESHOLD
from src.data_utils import _kuairec_dir


def load_raw(matrix: str, users: np.ndarray | None = None) -> pd.DataFrame:
    path = _kuairec_dir(DATA_RAW) / f"{matrix}.csv"
    df = pd.read_csv(
        path,
        usecols=["user_id", "video_id", "watch_ratio"],
        dtype={"user_id": np.int32, "video_id": np.int32, "watch_ratio": np.float32},
    )
    df = df.drop_duplicates(subset=["user_id", "video_id"])
    if users is not None:
        df = df[df["user_id"].isin(set(int(x) for x in users))]
    return df


def main() -> None:
    th = WATCH_RATIO_THRESHOLD
    print(f"正反馈阈值: watch_ratio > {th}\n")

    small = load_raw("small_matrix")
    dense_users = np.sort(small["user_id"].unique())
    print(f"随机曝光: {len(small):,} 条 | {len(dense_users):,} 用户 | {small['video_id'].nunique():,} 视频")

    big = load_raw("big_matrix", users=dense_users)
    print(f"常规曝光: {len(big):,} 条（仅这 {len(dense_users):,} 个用户）")

    small["pos"] = (small["watch_ratio"] > th).astype(np.int8)
    big["pos"] = (big["watch_ratio"] > th).astype(np.int8)

    a = big.groupby("user_id")["pos"].agg(["size", "sum"]).rename(columns={"size": "n_big", "sum": "p_big"})
    b = small.groupby("user_id")["pos"].agg(["size", "sum"]).rename(columns={"size": "n_small", "sum": "p_small"})
    df = a.join(b, how="inner").reset_index()

    df["rate_big"] = df["p_big"] / df["n_big"]
    df["rate_small"] = df["p_small"] / df["n_small"]
    df["lift"] = df["rate_big"] / df["rate_small"].replace(0, np.nan)

    n = len(df)
    print(f"\n参与配对比较的用户: {n:,}\n")

    print("=" * 74)
    print("配对比较：同一个用户，算法推荐 vs 随机曝光")
    print("=" * 74)
    print(f"  平均曝光条数        算法 {df['n_big'].mean():8.1f}   随机 {df['n_small'].mean():8.1f}")
    print(f"  平均有效率          算法 {df['rate_big'].mean():8.2%}   随机 {df['rate_small'].mean():8.2%}")
    print(f"  有效率中位数        算法 {df['rate_big'].median():8.2%}   随机 {df['rate_small'].median():8.2%}")
    print(f"  算法有效的用户占比  {(df['rate_big'] > df['rate_small']).mean():8.1%}")
    print(f"  平均增益            {df['rate_big'].mean() / df['rate_small'].mean():8.2f}×")

    # 配对显著性检验（有效率分布非正态，用 Wilcoxon 符号秩检验）
    stat, p = stats.wilcoxon(df["rate_big"], df["rate_small"])
    print(f"  Wilcoxon 配对检验   p = {p:.2e}  {'显著' if p < 0.001 else '不显著'}")

    print("\n" + "=" * 74)
    print("这组数字说明什么")
    print("=" * 74)
    rand_rate = df["rate_small"].mean()
    print(f"  1) 算法把有效率从 {rand_rate:.2%} 提到了 {df['rate_big'].mean():.2%}，"
          f"相当于 {df['rate_big'].mean() / rand_rate:.2f} 倍")
    print(f"  2) 随机投放时仍有 {rand_rate:.2%} 的概率被有效观看 —— 这部分是"
          f"「平台从没推过、但用户会喜欢」的内容")
    print(f"  3) 训练时若把未曝光物品当负样本，其中约 {rand_rate:.2%} 其实是正样本，"
          f"即每 {1 / rand_rate:.0f} 个负样本里就有 1 个标错")
    print(f"  4) 每个用户平均有 {df['p_small'].mean():.0f} 个「会喜欢但从未出现在信息流里」的视频")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "exposure_bias.csv", index=False)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_users": int(n),
        "threshold": th,
        "rate_algo": float(df["rate_big"].mean()),
        "rate_random": float(rand_rate),
        "lift": float(df["rate_big"].mean() / rand_rate),
        "share_users_algo_better": float((df["rate_big"] > df["rate_small"]).mean()),
        "wilcoxon_p": float(p),
        "mean_n_big": float(df["n_big"].mean()),
        "mean_n_small": float(df["n_small"].mean()),
        "mean_liked_unseen": float(df["p_small"].mean()),
    }
    with open(RESULTS / "exposure_bias.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {RESULTS / 'exposure_bias.csv'} 与 exposure_bias.json")


if __name__ == "__main__":
    main()
