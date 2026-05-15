"""离散时间对齐：时间戳最近邻配对与均匀重采样索引。"""

from __future__ import annotations

import numpy as np


def align_nearest_pairs(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    *,
    max_delta_sec: float,
) -> list[tuple[int, int]]:
    """
    对两路单调递增时间戳（秒）做贪心最近邻配对；超过 ``max_delta_sec`` 的配对丢弃。

    返回 ``(i, j)`` 列表，``i`` 为 ``ts_a`` 下标，``j`` 为 ``ts_b`` 下标。
    """
    ta = np.asarray(ts_a, dtype=np.float64).ravel()
    tb = np.asarray(ts_b, dtype=np.float64).ravel()
    if ta.size == 0 or tb.size == 0:
        return []

    pairs: list[tuple[int, int]] = []
    j = 0
    for i in range(ta.size):
        while j + 1 < tb.size and abs(tb[j + 1] - ta[i]) <= abs(tb[j] - ta[i]):
            j += 1
        if abs(tb[j] - ta[i]) <= float(max_delta_sec):
            pairs.append((int(i), int(j)))
    return pairs


def resample_index_map(n_source: int, n_target: int) -> np.ndarray:
    """将 ``n_source`` 个采样均匀映射到 ``n_target`` 个目标下标（含端点 clip）。"""
    if n_source <= 0 or n_target <= 0:
        return np.array([], dtype=np.int64)
    if n_target == 1:
        return np.array([0], dtype=np.int64)
    pos = np.linspace(0.0, n_source - 1.0, num=n_target)
    return np.clip(np.round(pos).astype(np.int64), 0, n_source - 1)
