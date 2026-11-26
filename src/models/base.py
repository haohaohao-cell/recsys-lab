"""模型基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "Recommender":
        ...

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """返回 (len(users), n_items) 的打分矩阵。"""
        ...
