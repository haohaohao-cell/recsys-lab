"""汇总所有模型的逐用户结果，生成统一的对比报表（Markdown）。

用法：python -m scripts.report
输出：results/report.md
"""
from __future__ import annotations

import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import group_by_item_popularity, group_by_user_activity, summarize
from src.split import leave_last_out

DISPLAY = {
    "Popular": "Popular",
    "ItemCF_k200": "ItemCF (k=200)",
    "ItemCF_k500": "ItemCF (k=500, α=0.7)",
    "BPR": "BPR-MF (d=64)",
    "SASRec": "SASRec",
    "semantic_category": "Semantic (类目)",
    "semantic_both": "Semantic (文案+类目)",
}


def main() -> None:
    inter, _ = load_dataset()
    split = leave_last_out(inter)
    gt = split.user_gt()

    rows, per_user = [], {}
    for f in sorted(RESULTS.glob("per_user_*.csv")):
        key = f.stem.replace("per_user_", "")
        name = DISPLAY.get(key, key)
        df = pd.read_csv(f)
        df = group_by_user_activity(df, split.train)
        df = group_by_item_popularity(df, split.train, gt, split.test)
        per_user[name] = df

        s = summarize(df, (10, 20))
        rows.append({"模型": name, **s})

    overall = pd.DataFrame(rows).sort_values("HR@10", ascending=False).set_index("模型")

    act = pd.DataFrame(
        {name: d.groupby("activity", observed=True)["hr@10"].mean() for name, d in per_user.items()}
    )
    act.index.name = "活跃度"
    item = pd.DataFrame(
        {name: d.groupby("item_bucket", observed=True)["hr@10"].mean() for name, d in per_user.items()}
    ).reindex(["长尾", "中部", "头部"])
    item.index.name = "物品分桶"

    def md(df: pd.DataFrame, index_name: str = "") -> str:
        idx_name = index_name or (df.index.name or "")
        cols = [idx_name] + list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for idx, r in df.iterrows():
            vals = [f"{v:.4f}" for v in r]
            out.append("| " + " | ".join([str(idx)] + vals) + " |")
        return "\n".join(out)

    text = (
        "# 实验结果汇总\n\n"
        "> 数据集：KuaiRec 短视频（watch_ratio > 2 视为有效观看）\n"
        "> 协议：leave-last-out 时序切分 + 全量排序（无负采样）\n\n"
        "## 总体指标\n\n"
        + md(overall, "模型")
        + "\n\n## 按用户活跃度分组（HR@10）\n\n"
        + md(act, "活跃度")
        + "\n\n## 按被预测物品流行度分组（HR@10）\n\n"
        + md(item, "物品分桶")
        + "\n"
    )

    (RESULTS / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"已写入 {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
