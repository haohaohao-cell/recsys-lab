"""不启动 Streamlit 也能检查看板依赖是否就绪。用法：python -m scripts.check_bundle"""
from __future__ import annotations

import time


def main() -> None:
    t0 = time.time()
    from src.artifacts import build_bundle, bucket_table, recommend

    b = build_bundle()
    print(f"加载耗时 {time.time() - t0:.1f}s")
    print(f"模型: {list(b.models)}")
    print(f"用户 {b.split.n_users:,} | 物品 {b.split.n_items:,}")

    u = int(next(iter(b.gt)))
    print(f"\n示例用户 u={u}，真实下一条 = {b.title[b.gt[u]][:28]!r}")
    for name in b.models:
        recs = recommend(b, u, name, 3)
        print(f"\n[{name}]")
        for r in recs:
            print(f"  #{r['rank']} {r['title'][:24]!r}  hit={r['hit']}  why={r['why']}")

    bt = bucket_table(b)
    print("\n分桶 HR@10:")
    print(bt.round(4).to_string())


if __name__ == "__main__":
    main()
