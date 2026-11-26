"""训练并评估 BPR-MF。用法：python -m scripts.train_bpr"""
from __future__ import annotations

import time

import pandas as pd

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import covered_items, evaluate, summarize
from src.models.bpr import BPR
from src.split import leave_last_out


def seen_from(df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for u, i in zip(df["u"].to_numpy(), df["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def main() -> None:
    inter, _ = load_dataset()
    split = leave_last_out(inter)

    val_gt = dict(zip(split.val["u"].to_numpy(), split.val["i"].to_numpy()))
    val_seen = seen_from(split.train)
    test_gt = split.user_gt()
    test_seen = seen_from(pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]]))

    model = BPR(dim=64, epochs=60)
    model.val_fn = lambda ep: float(
        summarize(
            evaluate(model.score, split.n_users, val_gt, val_seen, split.n_items, (10,), 256), (10,)
        )["HR@10"]
    )

    t0 = time.time()
    model.fit(split.train, split.n_users, split.n_items)
    print(f"训练耗时 {time.time() - t0:.1f}s")

    df = evaluate(model.score, split.n_users, test_gt, test_seen, split.n_items, (10, 20), 256)
    s = summarize(df, (10, 20))
    s["coverage@10"] = covered_items(model.score, split.n_users, split.n_items, test_seen, 10)
    print(f"\n[BPR-MF] HR@10={s['HR@10']:.4f}  NDCG@10={s['NDCG@10']:.4f}  "
          f"HR@20={s['HR@20']:.4f}  覆盖率@10={s['coverage@10']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "per_user_BPR.csv", index=False)
    pd.Series({**s, "model": model.name}).to_csv(RESULTS / "metrics_bpr.csv")


if __name__ == "__main__":
    main()
