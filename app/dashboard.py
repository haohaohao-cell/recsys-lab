"""RecLab · 推荐策略实验台

启动：
    streamlit run app/dashboard.py --server.port 8501
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RecLab · 推荐策略实验台", page_icon="🎬", layout="wide")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

STRATS = ["SASRec", "ItemCF", "Popular", "语义召回"]
COLOR = {"SASRec": "#6366f1", "ItemCF": "#0d9488", "Popular": "#f59e0b", "语义召回": "#e11d48"}
DESC = {
    "SASRec": "顺着最近看的往下猜",
    "ItemCF": "看了 A 的人也看了 B",
    "Popular": "全站最热",
    "语义召回": "内容/类目相近",
}

CSS = """
<style>
  .block-container { padding-top: 3.4rem; max-width: 1150px; }
  h1 { letter-spacing: -.02em; }

  .hero { font-size: 12px; color: #94a3b8; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 4px; }
  .sub { font-size: 14px; color: #64748b; margin: -6px 0 18px; }

  .kpi-row { display: flex; gap: 12px; margin: 6px 0 18px; }
  .kpi { flex: 1; background: #fff; border: 1px solid #e8edf3; border-radius: 14px; padding: 15px 17px;
         transition: box-shadow .15s ease, transform .15s ease; }
  .kpi:hover { box-shadow: 0 4px 16px rgba(15,23,42,.07); transform: translateY(-1px); }
  .kpi .v { font-size: 27px; font-weight: 700; color: #0f172a; line-height: 1.15; letter-spacing: -.02em;
            font-variant-numeric: tabular-nums; }
  .kpi .l { font-size: 12px; color: #475569; margin-top: 3px; font-weight: 500; }
  .kpi .s { font-size: 11px; color: #94a3b8; margin-top: 7px; }

  .rec-card { display: flex; align-items: flex-start; gap: 11px; background: #fff;
              border: 1px solid #e8edf3; border-radius: 12px; padding: 11px 14px; margin-bottom: 8px;
              transition: box-shadow .15s ease, transform .15s ease, border-color .15s ease; }
  .rec-card:hover { box-shadow: 0 4px 14px rgba(15,23,42,.08); transform: translateY(-1px); border-color: #cbd5e1; }
  .rec-card.hit { background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 60%); border-color: #bbf7d0; }

  .medal { width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center;
           justify-content: center; font-weight: 700; font-size: 13px; flex: none;
           background: #f1f5f9; color: #64748b; font-variant-numeric: tabular-nums; }
  .medal.m1 { background: #fef3c7; color: #b45309; }
  .medal.m2 { background: #e2e8f0; color: #475569; }
  .medal.m3 { background: #ffedd5; color: #c2410c; }

  .rec-main { flex: 1; min-width: 0; }
  .rec-title { font-size: 14px; font-weight: 600; color: #0f172a; line-height: 1.45;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .rec-meta { font-size: 11.5px; color: #64748b; margin-top: 3px; }
  .rec-why { font-size: 11.5px; color: #7c3aed; margin-top: 4px; }
  .cons { font-size: 10.5px; color: #94a3b8; margin-top: 5px; font-variant-numeric: tabular-nums; }
  .cons b { color: #64748b; font-weight: 600; }

  .flag { font-size: 10.5px; font-weight: 600; padding: 3px 9px; border-radius: 999px; flex: none; }
  .flag.hit { background: #dcfce7; color: #15803d; }
  .flag.miss { background: #f8fafc; color: #cbd5e1; }

  .hist-item { font-size: 12.5px; color: #334155; padding: 6px 0; border-bottom: 1px dashed #eef2f6; line-height: 1.5; }
  .hist-cat { color: #94a3b8; font-size: 10.5px; margin-top: 1px; }

  .gt-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 12px 14px; margin-top: 12px; }
  .gt-box .t { font-size: 10.5px; color: #b45309; letter-spacing: .08em; }
  .gt-box .n { font-size: 13px; color: #78350f; font-weight: 600; margin-top: 4px; line-height: 1.45; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
  .chip { font-size: 10.5px; background: #fff; border: 1px solid #fde68a; border-radius: 999px;
          padding: 2px 8px; color: #92400e; display: inline-flex; align-items: center; gap: 5px;
          font-variant-numeric: tabular-nums; }
  .chip i { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
  .chip.best { border-color: #86efac; background: #f0fdf4; color: #15803d; }

  .section { font-size: 14.5px; font-weight: 700; color: #0f172a; margin: 8px 0 10px;
             padding-left: 9px; border-left: 3px solid #6366f1; }
  .note { font-size: 12.5px; color: #64748b; line-height: 1.75; }
  .card { background: #fff; border: 1px solid #e8edf3; border-left: 3px solid #cbd5e1;
          border-radius: 12px; padding: 14px 16px; }
  .note b { color: #334155; }
  .pill-note { font-size: 11.5px; color: #64748b; background: #f8fafc; border-radius: 8px;
               padding: 8px 12px; margin-top: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- 数据加载
_BUNDLE = None


def get_bundle():
    """进程内单例。

    不用 st.cache_resource：会话被中断时它可能一直持有锁，把后续请求堵死。
    """
    global _BUNDLE
    if _BUNDLE is None:
        from src.artifacts import build_bundle

        _BUNDLE = build_bundle()
    return _BUNDLE


def load_json(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bucket_table(b):
    from src.artifacts import bucket_table as _bt

    return _bt(b)


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------- 图表
def _bars(df: pd.DataFrame, order: list[str], height: int = 280, colors: dict | None = None,
          color: str | None = None):
    """分类轴按显式顺序排列。

    Vega-Lite 的 nominal 轴默认按字母序排，会把「长尾/中部/头部」排乱，必须自己指定 sort。
    """
    import altair as alt

    long = df.reset_index()
    long.columns = ["类别"] + list(df.columns)
    long = long.melt(id_vars="类别", var_name="系列", value_name="值")

    base = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    x = alt.X("类别:N", sort=order, axis=alt.Axis(labelAngle=0, title=None, labelColor="#64748b"))
    y = alt.Y("值:Q", axis=alt.Axis(title=None, gridColor="#f1f5f9", labelColor="#94a3b8"))
    tip = [alt.Tooltip("类别:N"), alt.Tooltip("系列:N"), alt.Tooltip("值:Q", format=".4f")]

    if color is not None:
        return base.encode(x=x, y=y, color=alt.value(color), tooltip=tip).properties(height=height)
    enc = dict(x=x, y=y, color="系列:N", xOffset="系列:N", tooltip=tip)
    if colors:
        enc["color"] = alt.Color(
            "系列:N",
            scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        )
    return base.encode(**enc).properties(height=height)


# ---------------------------------------------------------------- 推荐与排名
def strategy_scores(b, u: int) -> dict[str, np.ndarray]:
    """一次算出四个策略对该用户的完整打分（已屏蔽已交互物品）。"""
    hidden = np.fromiter(b.seen.get(u, set()), dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for name in STRATS:
        if name not in b.models:
            continue
        s = np.asarray(b.models[name].score(np.array([u])), dtype=np.float32)[0].copy()
        if hidden.size:
            s[hidden] = -np.inf
        out[name] = s
    return out


def rank_of(scores: np.ndarray, item: int) -> int:
    return int((scores > scores[item]).sum()) + 1


def fmt_rank(r: int) -> str:
    return f"{r}" if r < 1000 else "999+"


def top_items(scores: np.ndarray, k: int) -> np.ndarray:
    from src.metrics import topk_indices

    return topk_indices(scores[None, :], k)[0]


# ---------------------------------------------------------------- 卡片
def card(rank: int, item: int, b, why: str, hit: bool, cons: list[tuple[str, int]]) -> str:
    medal = f"m{rank}" if rank <= 3 else ""
    flag = '<span class="flag hit">命中</span>' if hit else '<span class="flag miss">—</span>'
    # 只展示进入前 1000 的排名，否则满屏 999+ 反而看不清重点
    shown = [(n, r) for n, r in cons if r < 1000]
    if shown:
        cons_html = " · ".join(f"{n} <b>{fmt_rank(r)}</b>" for n, r in shown)
    else:
        cons_html = '<span style="color:#cbd5e1">其他策略均未进入前 1000</span>'
    return (
        f'<div class="rec-card{" hit" if hit else ""}">'
        f'<div class="medal {medal}">{rank}</div>'
        f'<div class="rec-main">'
        f'<div class="rec-title">{_esc(b.title[item])}</div>'
        f'<div class="rec-meta">{_esc(b.genres[item])}</div>'
        f'<div class="rec-why">↳ {_esc(why)}</div>'
        f'<div class="cons">其他策略排名：{cons_html}</div>'
        f"</div>{flag}</div>"
    )


def history_html(b, u: int, k: int = 7) -> str:
    rows = []
    for i in b.hist.get(u, [])[-k:][::-1]:
        rows.append(
            f'<div class="hist-item">{_esc(b.title[i][:34])}'
            f'<div class="hist-cat">{_esc(b.genres[i][:30])}</div></div>'
        )
    return "".join(rows)


def gt_panel(b, u: int, scores: dict[str, np.ndarray]) -> str:
    gt = b.gt.get(u)
    if gt is None:
        return ""
    ranks = [(n, rank_of(s, gt)) for n, s in scores.items()]
    best = min(r for _, r in ranks)
    chips = "".join(
        f'<span class="chip{" best" if r == best else ""}">'
        f'<i style="background:{COLOR[n]}"></i>{n} 第 {fmt_rank(r)}</span>'
        for n, r in ranks
    )
    return (
        f'<div class="gt-box"><div class="t">他接下来真的看了</div>'
        f'<div class="n">{_esc(b.title[gt][:40])}</div>'
        f'<div class="chips">{chips}</div></div>'
    )


def _goto(page: str) -> None:
    """按钮回调里改导航。

    必须在回调里改：脚本体内直接给 st.session_state.nav 赋值会因为
    「widget 已实例化」而抛异常，而回调在脚本重跑之前执行，是合法的。
    """
    st.session_state.nav = page


# ---------------------------------------------------------------- 概览
def page_overview(b) -> None:
    st.markdown('<div class="hero">KuaiRec · 快手短视频 · 离线实验</div>', unsafe_allow_html=True)
    st.markdown("# 推荐策略实验台")
    st.markdown(
        '<div class="sub">换一把尺子，冠军就会换人。这里交付的不是某一个模型，而是能守住结论的评测。</div>',
        unsafe_allow_html=True,
    )

    tbl = model_table()
    n = int(len(b.split.train) + len(b.split.val) + len(b.split.test))
    sasrec_hr = float(tbl.loc["SASRec", "HR@10"]) if "SASRec" in tbl.index else float("nan")
    bias = load_json("exposure_bias.json")
    lift = bias.get("lift", 0.0)
    naive = bias.get("rate_random", 0.0)
    tuned = bias.get("rate_algo", 0.0)

    bt = bucket_table(b)
    tail_nonzero = int((bt.loc["长尾"] > 0).sum()) if "长尾" in bt.index else 0
    n_models = int(bt.shape[1])

    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="v">{n:,}</div><div class="l">有效观看</div>'
        f'<div class="s">{b.split.n_users:,} 用户 · {b.split.n_items:,} 视频</div></div>'
        f'<div class="kpi"><div class="v">{sasrec_hr:.4f}</div><div class="l">SASRec HR@10</div>'
        f'<div class="s">严格时序 · 全量排序</div></div>'
        f'<div class="kpi"><div class="v">{lift:.2f}×</div><div class="l">算法 vs 随机投放</div>'
        f'<div class="s">{naive:.2%} → {tuned:.2%} 有效率</div></div>'
        f'<div class="kpi"><div class="v">{tail_nonzero} / {n_models}</div><div class="l">长尾非零模型</div>'
        f'<div class="s">两条都来自语义召回</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 4])
    c1.button("看用户案例 →", width="stretch", on_click=_goto, args=("用户视角",))
    c2.button("看诊断 →", width="stretch", on_click=_goto, args=("诊断",))

    st.markdown('<div class="section">模型排行榜 · next-item 严格时序</div>', unsafe_allow_html=True)
    st.altair_chart(_bars(tbl, list(tbl.index), height=250, color="#6366f1"), width="stretch")

    cols = st.columns(3)
    cards = [
        ("发现一 · 协议反转", "随机切分：ItemCF 0.0635 &gt; Popular 0.0478<br>"
                            "严格时序：ItemCF 0.0127 &lt; Popular 0.0141<br><br>"
                            "不只是数字高低变了，<b>冠军换人了</b>。"),
        ("发现二 · 任务反转", "next-item：SASRec 0.1416（第 1）<br>"
                            "稠密召回：ItemCF 0.1704（第 1）<br><br>"
                            "一个要精度，一个要覆盖，方向相反。"),
        ("发现三 · 长尾死区", "长尾桶 HR@10：协同过滤、BPR、SASRec <b>全是 0</b><br>"
                            f"只有语义召回 {bt.loc['长尾'].max():.4f}<br><br>"
                            "内容信号是长尾唯一的入口。"),
    ]
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="note">{body}</div>', unsafe_allow_html=True)


def model_table() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("per_user_*.csv")):
        key = f.stem.replace("per_user_", "")
        name = {
            "Popular": "Popular", "ItemCF_k200": "ItemCF", "ItemCF_k500": "ItemCF (k=500)",
            "BPR": "BPR-MF", "SASRec": "SASRec",
            "semantic_category": "语义召回(类目)", "semantic_both": "语义召回(文案)",
        }.get(key)
        if name is None:
            continue
        df = pd.read_csv(f, usecols=["hr@10"])
        rows.append({"模型": name, "HR@10": float(df["hr@10"].mean())})
    return pd.DataFrame(rows).sort_values("HR@10", ascending=False).set_index("模型")


# ---------------------------------------------------------------- 用户视角
def available_strats(b) -> list[str]:
    """当前环境实际可用的策略（云端没装 torch 时就没有 SASRec）。"""
    return [s for s in STRATS if s in b.models]


def page_user(b) -> None:
    from src.artifacts import explain

    strats = available_strats(b)
    idx2user = {v: k for k, v in b.split.user2idx.items()}
    if "uid" not in st.session_state:
        st.session_state.uid = _find_demo_user(b)
    st.session_state.setdefault("strat", strats[0])
    if st.session_state.strat not in strats:
        st.session_state.strat = strats[0]
    st.session_state.setdefault("topk", 10)

    u = int(st.session_state.uid)
    scores = strategy_scores(b, u)

    left, right = st.columns([3, 7], gap="large")

    with left:
        st.markdown('<div class="section">用户</div>', unsafe_allow_html=True)
        st.markdown(
            f"**用户 #{idx2user.get(u, u)}** · 有效观看 {len(b.hist.get(u, []))} 条", unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        if c1.button("换一个", width="stretch"):
            st.session_state.uid = int(np.random.default_rng().choice(list(b.gt.keys())))
            st.rerun()
        if c2.button("找个命中的", width="stretch"):
            st.session_state.uid = _find_demo_user(b, exclude=u)
            st.rerun()

        st.markdown('<div class="section" style="margin-top:16px">最近看过</div>', unsafe_allow_html=True)
        st.markdown(history_html(b, u), unsafe_allow_html=True)
        st.markdown(gt_panel(b, u, scores), unsafe_allow_html=True)

    with right:
        compare = st.toggle("并排对比 · SASRec vs ItemCF", value=False)

        if not compare:
            strat = st.segmented_control(
                "策略", strats, key="strat", label_visibility="collapsed",
            ) or strats[0]
            st.slider("推荐条数", 5, 20, step=5, key="topk")

            s = scores[strat]
            k = st.session_state.topk
            top = top_items(s, k)
            gt = b.gt.get(u)

            hit = int((top == gt).sum())
            st.markdown(
                f'<div class="pill-note">策略 <b>{strat}</b> · {DESC[strat]} ｜ '
                f'top-{k} 命中 <b>{hit}</b> 条 ｜ '
                f'{"✅ 命中真实下一条" if hit else "❌ 未命中"}</div>',
                unsafe_allow_html=True,
            )

            html = []
            for rank, item in enumerate(top, 1):
                cons = [(n, rank_of(scores[n], int(item))) for n in strats if n != strat and n in scores]
                html.append(
                    card(rank, int(item), b, explain(b, strat, u, int(item)), int(item) == gt, cons)
                )
            st.markdown("".join(html), unsafe_allow_html=True)
        else:
            k = 5
            gt = b.gt.get(u)
            pair = [s for s in ("SASRec", "ItemCF") if s in strats] or strats[:2]
            c1, c2 = st.columns(2, gap="medium")
            for col, strat in zip((c1, c2), pair):
                with col:
                    top = top_items(scores[strat], k)
                    hit = int((top == gt).sum())
                    st.markdown(
                        f'<div class="section" style="border-left-color:{COLOR[strat]}">'
                        f'{strat} · 命中 {hit}</div>',
                        unsafe_allow_html=True,
                    )
                    html = []
                    for rank, item in enumerate(top, 1):
                        cons = [(n, rank_of(scores[n], int(item))) for n in ("Popular", "语义召回")
                                if n in scores]
                        html.append(
                            card(rank, int(item), b, explain(b, strat, u, int(item)),
                                 int(item) == gt, cons)
                        )
                    st.markdown("".join(html), unsafe_allow_html=True)

    st.caption(
        "卡片底部的「其他策略排名」是这个视频在别的策略里排第几——同一个视频，"
        "SASRec 可能排第 1，Popular 可能排到几百名开外。命中的卡片会变绿。"
    )


def _find_demo_user(b, exclude: int | None = None) -> int:
    """找一个首屏就有命中的用户，让演示一开始有东西可看。"""
    from src.artifacts import recommend

    strats = available_strats(b)
    for u in list(b.gt.keys())[:200]:
        if u == exclude:
            continue
        for s in strats:
            if any(r["hit"] for r in recommend(b, u, s, 3)):
                return int(u)
    return int(next(iter(b.gt)))


# ---------------------------------------------------------------- 错例分析
@st.cache_data(show_spinner=False)
def failure_table() -> pd.DataFrame:
    p = RESULTS / "failure_ranks.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def page_failure(b) -> None:
    st.markdown('<div class="hero">错例分析</div>', unsafe_allow_html=True)
    st.markdown("# 模型在哪里彻底失效")

    df = failure_table()
    if df.empty:
        st.warning("缺少 results/failure_ranks.csv，先运行 `python -m scripts.failure_analysis`。")
        return

    n = len(df)
    fail = df[df["all_fail"]]
    near = fail[(fail["best_rank"] > 10) & (fail["best_rank"] <= 50)]
    far = fail[fail["best_rank"] > 1000]
    ok = df[~df["all_fail"]]
    ratio = ok["gt_pop"].median() / max(fail["gt_pop"].median(), 1)

    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="v">{len(fail) / n:.1%}</div><div class="l">四个模型全部落空</div>'
        f'<div class="s">{len(fail):,} / {n:,} 个测试用户</div></div>'
        f'<div class="kpi"><div class="v">{len(near) / max(len(fail), 1):.1%}</div>'
        f'<div class="l">只是「差一点」</div>'
        f'<div class="s">最佳排名 11–50，排序优化可救</div></div>'
        f'<div class="kpi"><div class="v">{len(far) / max(len(fail), 1):.1%}</div>'
        f'<div class="l">完全够不到</div>'
        f'<div class="s">最佳排名 &gt; 1000，信号不存在</div></div>'
        f'<div class="kpi"><div class="v">{ratio:.1f}×</div><div class="l">命中组的物品更热门</div>'
        f'<div class="s">流行度中位 {ok["gt_pop"].median():.0f} vs {fail["gt_pop"].median():.0f}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section">几个模型同时命中</div>', unsafe_allow_html=True)
        cnt = df["n_hit"].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
        st.altair_chart(
            _bars(pd.DataFrame({"用户数": cnt.to_numpy()}, index=[str(i) for i in cnt.index]),
                  [str(i) for i in cnt.index], height=230, color="#e11d48"),
            width="stretch",
        )
        st.markdown(
            '<div class="note">绝大多数用户是 0 个模型命中。'
            "四个模型同时命中的情况<b>一次都没出现</b>——它们的强项几乎不重叠。</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section">真实下一条有多热门</div>', unsafe_allow_html=True)
        bins = [0, 2, 10, 50, 200, 10**9]
        labels = ["0–2", "3–10", "11–50", "51–200", "200+"]
        f_c = pd.cut(fail["gt_pop"], bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
        o_c = pd.cut(ok["gt_pop"], bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
        tbl = pd.DataFrame(
            {"全部落空": (f_c / max(len(fail), 1)).to_numpy(),
             "至少一个命中": (o_c / max(len(ok), 1)).to_numpy()},
            index=labels,
        )
        st.altair_chart(
            _bars(tbl, labels, height=230,
                  colors={"全部落空": "#fca5a5", "至少一个命中": "#10b981"}),
            width="stretch",
        )
        st.markdown(
            f'<div class="note">失败组的真实下一条中位只被看过 <b>{fail["gt_pop"].median():.0f}</b> 次，'
            f"命中组是 <b>{ok['gt_pop'].median():.0f}</b> 次——差 {ratio:.1f} 倍。<br><br>"
            "但注意失败组的主力<b>不是「冷到没人看」</b>（≤5 次的只占 "
            f"{(fail['gt_pop'] <= 5).mean():.1%}），而是 51–200 次这段「中等热门」："
            "有曝光、够进候选集，但不足以让协同信号稳定。</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section">翻一个失败案例看看</div>', unsafe_allow_html=True)

    idx2user = {v: k for k, v in b.split.user2idx.items()}
    fail_users = fail["user"].to_numpy()
    if "fail_uid" not in st.session_state or st.session_state.fail_uid not in set(fail_users.tolist()):
        st.session_state.fail_uid = int(fail_users[0])

    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("换一个失败案例", width="stretch",
              on_click=lambda: st.session_state.__setitem__(
                  "fail_uid", int(np.random.default_rng().choice(fail_users))))
    c2.button("看一个「差一点」的", width="stretch",
              on_click=lambda: st.session_state.__setitem__(
                  "fail_uid", int(np.random.default_rng().choice(near["user"].to_numpy()))))

    u = int(st.session_state.fail_uid)
    row = df[df["user"] == u].iloc[0]
    gt = b.gt.get(u)

    cl, cr = st.columns([3, 7], gap="large")
    with cl:
        st.markdown(
            f"**用户 #{idx2user.get(u, u)}** · 有效观看 {len(b.hist.get(u, []))} 条",
            unsafe_allow_html=True,
        )
        st.markdown(history_html(b, u, 6), unsafe_allow_html=True)
        if gt is not None:
            ranks = [(s, int(row[f"rank_{s}"])) for s in ["SASRec", "ItemCF", "Popular", "Semantic"]]
            best = min(r for _, r in ranks)
            chips = "".join(
                f'<span class="chip{" best" if r == best else ""}">'
                f'<i style="background:{COLOR.get(s, "#94a3b8")}"></i>'
                f'{s} 第 {fmt_rank(r)}</span>'
                for s, r in ranks
            )
            st.markdown(
                f'<div class="gt-box"><div class="t">他接下来真的看了</div>'
                f'<div class="n">{_esc(b.title[gt][:44])}</div>'
                f'<div class="chips">{chips}</div></div>',
                unsafe_allow_html=True,
            )

    with cr:
        br = int(row["best_rank"])
        if br <= 50:
            verdict, tone = (
                "差一点——最佳排名只有 "
                f"{br}，说明信号是有的，缺的是一个更好的排序器把它的分数再抬一点。", "#f59e0b"
            )
        elif br <= 1000:
            verdict, tone = (
                f"中等距离——最佳排名 {br}，在 10,610 个候选里已经算出了一些倾向，但证据太弱。", "#0ea5e9"
            )
        else:
            verdict, tone = (
                "完全够不到——四个模型都排到 1000 名开外。这个视频在训练集里只出现 "
                f"{int(row['gt_pop'])} 次，现有特征里根本没有它的信号。", "#ef4444"
            )

        st.markdown(
            f'<div class="card" style="border-left-color:{tone}">'
            f'<div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:6px">诊断</div>'
            f'<div style="font-size:12.5px;color:#475569;line-height:1.7">{verdict}</div>'
            f'<div style="font-size:11.5px;color:#94a3b8;margin-top:10px">'
            f'用户历史 {len(b.hist.get(u, []))} 条 · 真实物品训练集出现 {int(row["gt_pop"])} 次 · '
            f'候选池 {b.split.n_items:,}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="note" style="margin-top:14px">'
            "这一类案例的价值在于把「失败」拆成两半：<b>差一点的</b>是排序问题，工程上可解；"
            "<b>够不到的</b>是数据问题，只能等这个物品积累足够的交互，或者引入内容/外部信号。"
            "混淆这两者，就会在错误的方向上使劲。</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- 诊断
def page_diag(b) -> None:
    st.markdown('<div class="hero">实验诊断</div>', unsafe_allow_html=True)
    st.markdown("# 三张图说清结论")

    t1, t2, t3, t4 = st.tabs(["① 协议", "② 任务", "③ 长尾", "④ 融合"])

    with t1:
        p = RESULTS / "protocol_compare.csv"
        if p.exists():
            df = pd.read_csv(p)
            piv = df.pivot(index="模型", columns="协议", values="HR@10")
            st.altair_chart(
                _bars(piv, ["随机留一（泄漏未来）", "时序留最后（严格）"], height=300),
                width="stretch",
            )
        st.markdown(
            '<div class="note">深色柱是严格时序，浅色柱是随机切分。<br>'
            "随机切分下 ItemCF（0.0635）明显高于 Popular（0.0478）；"
            "换成严格时序，ItemCF（0.0127）反而低于 Popular（0.0141）——<b>冠军反转</b>。<br><br>"
            "个性化模型虚高得更厉害（5.0× vs 3.4×），所以宽松协议会系统性夸大"
            "「个性化」的价值。</div>",
            unsafe_allow_html=True,
        )

    with t2:
        p = RESULTS / "dense_eval.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0).sort_values("Recall@20", ascending=False)
            st.altair_chart(
                _bars(df[["Recall@20"]], list(df.index), height=300, color="#0ea5e9"),
                width="stretch",
            )
        st.markdown(
            '<div class="note">稠密矩阵上每个用户有约 154 个真实喜欢的视频，衡量的是覆盖面。'
            "SASRec 从第 1 掉到第 3——它把概率压得太尖，适合猜下一条，不适合覆盖兴趣。<br><br>"
            "所以「哪个模型最好」这句话，不指定任务就没有答案。</div>",
            unsafe_allow_html=True,
        )

    with t3:
        bt = bucket_table(b)
        bt.columns = [
            {"Popular": "Popular", "ItemCF_k200": "ItemCF", "BPR": "BPR-MF", "SASRec": "SASRec",
             "semantic_category": "语义召回(类目)", "semantic_both": "语义召回(文案)"}.get(c, c)
            for c in bt.columns
        ]
        keep = [c for c in ["Popular", "ItemCF", "BPR-MF", "SASRec", "语义召回(类目)"] if c in bt.columns]
        st.altair_chart(_bars(bt[keep], ["长尾", "中部", "头部"], height=320), width="stretch")
        st.markdown(
            '<div class="note">「长尾」这一组里，除语义召回外全部为 0。'
            "协同过滤需要共现证据，长尾视频没有；内容信号不需要。<br><br>"
            "这正是多路召回存在的理由——不是取平均，而是<b>按场景路由</b>。</div>",
            unsafe_allow_html=True,
        )

    with t4:
        p = RESULTS / "metrics_fusion.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0).rename(columns={"0": "整体命中率"})
            df["整体命中率"] = df["整体命中率"].map(lambda x: f"{x:.4f}")
            st.dataframe(df, width="stretch")
        st.markdown(
            '<div class="note">候选预算固定 130：全给序列模型最强（0.4747）。<br>'
            "分 30 个名额给语义召回，长尾命中率 +21.6%，但整体掉 4.08 个百分点——"
            "<b>预算守恒</b>，分给长尾的名额本来能捞到头部。<br><br>"
            "结论：只考核整体命中率时不该加；长尾有独立考核目标时才值得。</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- 数据体检
def page_data(b) -> None:
    import altair as alt

    st.markdown('<div class="hero">数据体检</div>', unsafe_allow_html=True)
    st.markdown("# 这份数据有什么毛病")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section">物品流行度长尾</div>', unsafe_allow_html=True)
        s = np.sort(b.pop)[::-1]
        cum = np.cumsum(s) / s.sum()
        df = pd.DataFrame({"排名": np.arange(1, len(cum) + 1), "累计占比": cum})
        step = max(len(df) // 250, 1)
        area = (
            alt.Chart(df.iloc[::step])
            .mark_area(line=True, color="#6366f1", opacity=0.25)
            .encode(
                x=alt.X("排名:Q", axis=alt.Axis(title=None)),
                y=alt.Y("累计占比:Q", axis=alt.Axis(format="%", title=None)),
                tooltip=[alt.Tooltip("排名:Q", format=","), alt.Tooltip("累计占比:Q", format=".1%")],
            )
            .properties(height=230)
        )
        st.altair_chart(area, width="stretch")
        top10 = cum[max(int(len(s) * 0.1) - 1, 0)]
        st.caption(
            f"前 10% 的视频吃掉了 {top10:.0%} 的有效观看，后 50% 合计不足 5%。"
            f"最热的一个占 {s[0] / s.sum():.1%}。"
        )

    with right:
        st.markdown('<div class="section">相邻两次有效观看的间隔</div>', unsafe_allow_html=True)
        gap = _gap_hours(b)
        hist, edges = np.histogram(gap, bins=np.arange(0, 73, 3))
        labels = [f"{int(e)}h" for e in edges[:-1]]
        st.altair_chart(
            _bars(pd.DataFrame({"用户数": hist}, index=labels), labels, height=230, color="#f59e0b"),
            width="stretch",
        )
        st.caption(
            f"间隔中位数 {np.median(gap):.0f} 小时，1 小时内连续观看只占 {(gap < 1).mean():.1%}。"
            "「看了 A 的人也看了 B」在短视频上不成立——这是 ItemCF 失效的根因。"
        )

    st.markdown('<div class="section">稠密子集 · 可以用来做无偏评估</div>', unsafe_allow_html=True)
    d = b.dense_stats
    bias = load_json("exposure_bias.json")
    cols = st.columns(4)
    items = [
        (f"{d['coverage']:.2%}", "稠密矩阵覆盖率", f"{d['n_users']:,} 用户 × {d['n_items']:,} 视频"),
        (f"{d['pos_per_user']:.0f}", "人均真实喜欢", "watch_ratio > 2"),
        (f"{bias.get('rate_random', 0):.2%}", "随机曝光有效率", f"算法为 {bias.get('rate_algo', 0):.2%}"),
        (f"{bias.get('mean_n_big', 0) / b.split.n_items:.1%}", "常规曝光覆盖率",
         f"人均 {bias.get('mean_n_big', 0):.0f} 条 / {b.split.n_items:,} 视频"),
    ]
    for col, (v, l, s2) in zip(cols, items):
        col.markdown(
            f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div>'
            f'<div class="s">{s2}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="note" style="margin-top:14px">小矩阵覆盖了 1,411 个用户对 3,327 个视频的 99.62%，'
        "而且这些视频<b>刻意避开了用户的常规信息流</b>——所以它是一份随机曝光实验，"
        "能回答「如果推给他，他会不会看」，这是普通数据集做不到的。</div>",
        unsafe_allow_html=True,
    )


def _gap_hours(b) -> np.ndarray:
    sp = b.split
    last = sp.test.set_index("u")["timestamp"]
    prev = sp.train.groupby("u")["timestamp"].max()
    return ((last - prev.reindex(last.index)) / 3600).to_numpy()


# ---------------------------------------------------------------- 入口
def main() -> None:
    try:
        with st.spinner("首次加载模型（约 30 秒，之后 5 秒）..."):
            b = get_bundle()
    except FileNotFoundError:
        st.error("**未找到 KuaiRec 数据**。首次运行需要先下载（约 412 MB，支持断点续传）。")
        st.code("python -m scripts.download_kuairec", language="bash")
        st.caption("下载完成后重新运行 `streamlit run app/dashboard.py` 即可。")
        st.stop()

    pages = ["概览", "用户视角", "错例分析", "诊断", "数据体检"]
    slug = {"overview": "概览", "user": "用户视角", "fail": "错例分析",
            "diag": "诊断", "data": "数据体检"}

    if "nav" not in st.session_state:
        st.session_state.nav = slug.get(st.query_params.get("page", ""), "概览")

    with st.sidebar:
        st.markdown("### 🎬 RecLab")
        st.caption("推荐策略实验台")
        page = st.radio("导航", pages, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("数据集 KuaiRec（快手）")
        st.caption(f"{b.split.n_users:,} 用户 · {b.split.n_items:,} 视频")
        st.caption("协议 严格时序 + 全量排序")
        st.divider()
        st.caption("四个策略各有一个颜色，贯穿全站：")
        for name in STRATS:
            st.markdown(
                f'<div style="font-size:12px;color:#475569;margin:3px 0">'
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{COLOR[name]};margin-right:7px"></span>{name}'
                f'<span style="color:#94a3b8;font-size:11px"> · {DESC[name]}</span></div>',
                unsafe_allow_html=True,
            )

    if page == "概览":
        page_overview(b)
    elif page == "用户视角":
        page_user(b)
    elif page == "错例分析":
        page_failure(b)
    elif page == "诊断":
        page_diag(b)
    else:
        page_data(b)


main()
