"""训练并评估 SASRec。

用法：
    python -m scripts.train_sasrec                # 正式训练
    python -m scripts.train_sasrec --smoke        # 冒烟测试（2 epoch，小规模）
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.config import RESULTS
from src.data_utils import load_dataset
from src.metrics import evaluate, summarize
from src.models.sasrec import SASRec
from src.split import leave_last_out


def seen_from(df: pd.DataFrame) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = {}
    for u, i in zip(df["u"].to_numpy(), df["i"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"device = {'cuda:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    inter, _ = load_dataset()
    split = leave_last_out(inter)

    val_gt = dict(zip(split.val["u"].to_numpy(), split.val["i"].to_numpy()))
    val_seen = seen_from(split.train)
    test_gt = split.user_gt()
    test_seen = seen_from(pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]]))

    model = SASRec(
        hidden=args.hidden,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=2 if args.smoke else args.epochs,
        patience=args.patience,
    )

    def val_metric(epoch: int) -> float:
        model.set_history(split.train, split.n_users, split.n_items)
        df = evaluate(model.score, split.n_users, val_gt, val_seen, split.n_items, (10,), 256)
        return float(summarize(df, (10,))["HR@10"])

    model.val_fn = val_metric
    t0 = time.time()
    model.fit(split.train, split.n_users, split.n_items)
    print(f"训练耗时 {time.time() - t0:.1f}s")

    # 测试：历史 = train + val（与 ItemCF 的协议一致，模型没见过 val）
    hist = pd.concat([split.train[["u", "i", "timestamp"]], split.val[["u", "i", "timestamp"]]])
    model.set_history(hist, split.n_users, split.n_items)
    df = evaluate(model.score, split.n_users, test_gt, test_seen, split.n_items, (10, 20), 256)
    s = summarize(df, (10, 20))
    print(f"\n[SASRec] 测试集 HR@10={s['HR@10']:.4f}  NDCG@10={s['NDCG@10']:.4f}  "
          f"HR@20={s['HR@20']:.4f}  NDCG@20={s['NDCG@20']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "per_user_SASRec.csv", index=False)
    if not args.smoke:
        torch.save(model.net.state_dict(), RESULTS / "sasrec.pt")
        pd.Series({**s, "model": model.name}).to_csv(RESULTS / "metrics_sasrec.csv")


if __name__ == "__main__":
    main()
