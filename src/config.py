"""全局配置：路径、数据协议、评估参数。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 数据默认放在仓库内的 data/ 下；可用环境变量 RECLAB_DATA 指向别处
# （本机 data/ 是一个指向 D 盘的目录联接，避免占用系统盘）
DATA_ROOT = Path(os.environ.get("RECLAB_DATA", ROOT / "data"))
DATA_RAW = DATA_ROOT / "raw"
DATA_PROC = DATA_ROOT / "processed"
RESULTS = ROOT / "results"

# 精简数据包：从原始数据导出的小体积 parquet，随仓库一起提交。
# 云端部署时没有 412MB 原始数据，直接读这里就能把看板跑起来。
ASSETS = ROOT / "assets"
DEMO_DIR = ASSETS / "demo"

DEFAULT_DATASET = "kuairec"

# ---------------- 数据协议 ----------------
# MovieLens：评分 >= 4 视为正反馈
MIN_RATING = 4.0
# 通用：用户最少交互数，不足则丢弃（保证 train/val/test 各至少 1 条）
MIN_INTERACTIONS = 5

# KuaiRec：watch_ratio（播放时长/视频时长）> 2 视为有效观看
# 这是 KuaiRec 论文与多数后续工作的惯例：有效观看 ≈ 看完两遍以上
WATCH_RATIO_THRESHOLD = 2.0

# ---------------- 评估协议 ----------------
KS = (10, 20)          # 截断位置
EVAL_BATCH = 256       # 每批评估的用户数
# 全量排序（不做负采样）。若改为采样评估，需保证所有模型共用同一份负样本。
