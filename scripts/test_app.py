"""用 Streamlit AppTest 逐页跑一遍看板，捕获异常。

用法：python -m scripts.test_app
"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def main() -> None:
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180)
    at.run()

    if at.exception:
        _fail(f"首页异常：{at.exception[0].value}")
    print(f"✅ 概览页渲染成功（{len(at.markdown)} 个 markdown 块）")

    for page in ["用户视角", "错例分析", "诊断", "数据体检"]:
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            _fail(f"{page} 异常：{at.exception[0].value}")
        print(f"✅ {page} 渲染成功")

    at.sidebar.radio[0].set_value("用户视角").run()
    for strat in ["ItemCF", "Popular", "语义召回", "SASRec"]:
        at.session_state["strat"] = strat
        at.run()
        if at.exception:
            _fail(f"策略 {strat} 异常：{at.exception[0].value}")
        print(f"✅ 策略切换 {strat} 正常")

    for k in (5, 20):
        at.session_state["topk"] = k
        at.run()
        if at.exception:
            _fail(f"top-{k} 异常：{at.exception[0].value}")
        print(f"✅ 推荐条数 {k} 正常")

    # 换用户
    btn = [b for b in at.button if "换一个" in str(b.label)]
    if btn:
        btn[0].click().run()
        if at.exception:
            _fail(f"换用户异常：{at.exception[0].value}")
        print("✅ 换用户正常")

    print("\n全部通过")


if __name__ == "__main__":
    main()
