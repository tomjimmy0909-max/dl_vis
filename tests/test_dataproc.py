"""logic.dataproc 轻量单测。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dl_vis.logic.dataproc import (
    DataProcConfig,
    align_nearest_pairs,
    process_path_to_nchw,
    resample_index_map,
    split_indices_train_val_test,
)
from dl_vis.logic.dataproc.mel_spectrogram import waveform_to_mel_nchw


class TestDataprocSplit(unittest.TestCase):
    def test_split_covers_all(self) -> None:
        tr, va, te = split_indices_train_val_test(100, 0.7, 0.15, 0.15, seed=123)
        self.assertEqual(len(tr) + len(va) + len(te), 100)
        self.assertEqual(len(set(tr) | set(va) | set(te)), 100)

    def test_split_ratio_error(self) -> None:
        with self.assertRaises(ValueError):
            split_indices_train_val_test(10, 0.5, 0.5, 0.5, seed=0)


class TestDataprocAlign(unittest.TestCase):
    def test_resample_map(self) -> None:
        m = resample_index_map(10, 5)
        self.assertEqual(m.shape, (5,))
        self.assertTrue(m[0] == 0 and m[-1] == 9)

    def test_nearest_pairs(self) -> None:
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([0.05, 1.1, 2.2])
        p = align_nearest_pairs(a, b, max_delta_sec=0.2)
        self.assertEqual(len(p), 3)


class TestDataprocMel(unittest.TestCase):
    def test_mel_shape(self) -> None:
        w = torch.randn(8000)
        m = waveform_to_mel_nchw(
            w,
            16000,
            n_fft=512,
            hop_length=128,
            n_mels=32,
            f_min=0.0,
            f_max=8000.0,
            target_height=32,
            target_width=100,
            log_floor=1e-6,
        )
        self.assertEqual(m.shape, (1, 32, 100))


class TestDataprocImagePipeline(unittest.TestCase):
    def test_png_roundtrip(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no pillow")
        cfg = DataProcConfig(
            modality="image",
            target_height=32,
            target_width=48,
            use_hist_equalize=False,
            normalize_mode="zero_one",
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            p = Path(f.name)
        try:
            Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(p)
            t = process_path_to_nchw(p, cfg)
            self.assertEqual(tuple(t.shape), (3, 32, 48))
            self.assertEqual(t.dtype, torch.float32)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
