"""正确性自检：用「作弊模型」和人工可验算的小例子，验证评估与 ItemCF 实现。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_utils import load_dataset
from src.metrics import evaluate
from src.models.itemcf import ItemCF
from src.split import leave_last_out

# ---------- 1. 作弊模型：直接把答案放到第 1 名，HR@K 必须等于 1 ----------
inter, _ = load_dataset()
split = leave_last_out(inter)
gt = split.user_gt()
seen = {}
both = pd.concat([split.train[["u", "i"]], split.val[["u", "i"]]])
for u, i in zip(both["u"].to_numpy(), both["i"].to_numpy()):
    seen.setdefault(int(u), set()).add(int(i))


class Oracle:
    name = "Oracle"

    def score(self, users):
        out = np.zeros((len(users), split.n_items), dtype=np.float32)
        for r, u in enumerate(users):
            out[r, gt[int(u)]] = 1.0
        return out


df = evaluate(Oracle().score, split.n_users, gt, seen, split.n_items, (10,), 256)
print(f"[1] 作弊模型 HR@10 = {df['hr@10'].mean():.4f}  (期望 1.0000)  通过={df['hr@10'].mean() == 1.0}")

# ---------- 2. 人工可验算的 ItemCF ----------
# 用户: u0={a,b}, u1={a,c}, u2={b,c}, u3={a,b,c}
# 物品共现: (a,b)=2, (a,c)=2, (b,c)=2 -> 余弦相似度 = 2/sqrt(3*3) = 2/3
train = pd.DataFrame(
    {"u": [0, 0, 1, 1, 2, 2, 3, 3, 3], "i": [0, 1, 0, 2, 1, 2, 0, 1, 2], "timestamp": 0}
)
m = ItemCF(k_neighbors=10, alpha=0.5).fit(train, n_users=4, n_items=3)
expected = np.array([[0, 2 / 3, 2 / 3], [2 / 3, 0, 2 / 3], [2 / 3, 2 / 3, 0]], dtype=np.float32)
print(f"[2] 相似度矩阵正确 = {np.allclose(m.S, expected, atol=1e-5)}")

sc = m.score(np.array([0, 1, 2, 3]))
ok_score = abs(sc[0, 2] - 4 / 3) < 1e-5 and abs(sc[0, 0] - 2 / 3) < 1e-5
print(f"[3] 打分正确 = {ok_score}  (u0 对 c 的分数 = {sc[0, 2]:.4f}, 期望 {4/3:.4f})")

# ---------- 3. 候选集完整性 ----------
bad = sum(1 for u, i in gt.items() if i in seen.get(u, set()))
print(f"[4] 被误 mask 的 GT 数量 = {bad}  (期望 0)  通过={bad == 0}")

# ---------- 4. 随机打分应接近理论下界 K/n_items ----------
rng = np.random.default_rng(0)


class Rand:
    name = "Random"

    def score(self, users):
        return rng.random((len(users), split.n_items)).astype(np.float32)


dfr = evaluate(Rand().score, split.n_users, gt, seen, split.n_items, (10,), 256)
theory = 10 / split.n_items
print(f"[5] 随机模型 HR@10 = {dfr['hr@10'].mean():.4f}  (理论 ~{theory:.4f})")
