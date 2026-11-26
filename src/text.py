"""物品文本处理：清洗与拼接。

KuaiRec 的文案噪声很大——63.6% 含 UNKNOWN，还有大量
`#感谢快手我要上热门` 这类出现在几乎每个视频里的模板标签。
不清洗的话，任何基于文本的模型都会把所有视频编码成几乎相同的向量。
"""
from __future__ import annotations

import re

import pandas as pd

_HASH = re.compile(r"#[^\s#]+")
_MENTION = re.compile(r"@[^\s@]+")
_SPACE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = _HASH.sub(" ", str(s))
    s = _MENTION.sub(" ", s)
    s = s.replace("UNKNOWN", " ")
    return _SPACE.sub(" ", s).strip()


def build_texts(items: pd.DataFrame, split, source: str = "both") -> list[str]:
    """按内部物品下标排列的文本。

    source: caption（只看文案）/ category（只看类目）/ both（两者拼接）
    """
    idx2item = {v: k for k, v in split.item2idx.items()}
    cat = items["genres"].astype(str).str.replace("|", " ", regex=False)
    if source == "category":
        text = cat
    elif source == "caption":
        text = items["title"].map(clean_text)
    else:
        text = items["title"].map(clean_text) + " " + cat
    by_id = {int(i): str(t) for i, t in zip(items["item_id"], text)}
    return [by_id.get(idx2item[i], "") for i in range(split.n_items)]


def display_title(t: str, limit: int = 34) -> str:
    """卡片上显示的标题：清洗 + 截断。"""
    t = clean_text(t)
    if not t:
        return "（无文案）"
    return t if len(t) <= limit else t[: limit - 1] + "…"
