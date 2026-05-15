"""训练 / 验证 / 测试索引划分。"""

from __future__ import annotations

import math

import numpy as np


def split_indices_train_val_test(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回打乱的索引数组 ``train_idx, val_idx, test_idx``，三者无交且并集为 ``0..n-1``。"""
    if n <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    r_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(r_sum - 1.0) > 1e-5:
        raise ValueError(f"train/val/test 比例之和应为 1，当前为 {r_sum}")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_tr = min(n, int(math.floor(n * float(train_ratio))))
    n_va = min(max(0, n - n_tr), int(math.floor(n * float(val_ratio))))
    n_te = n - n_tr - n_va
    train_idx = idx[:n_tr]
    val_idx = idx[n_tr : n_tr + n_va]
    test_idx = idx[n_tr + n_va :]
    return train_idx, val_idx, test_idx
