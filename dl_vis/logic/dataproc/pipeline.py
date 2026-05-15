"""按路径与配置调度各模态，输出 ``(C,H,W)`` float32。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dl_vis.logic.dataproc.config import DataProcConfig
from dl_vis.logic.dataproc.image_prep import chw_uint8_to_tensor_prep
from dl_vis.logic.dataproc.mel_spectrogram import mel_spectrogram_file
from dl_vis.logic.dataproc.video_frames import video_clip_to_nchw


def load_image_chw_uint8(path: str | Path) -> torch.Tensor:
    """``uint8`` 张量 ``(C,H,W)``，``C`` 为 1 或 3。"""
    path = Path(path)
    try:
        from PIL import Image

        im = Image.open(path)
        if im.mode in ("L", "1"):
            arr = np.asarray(im.convert("L"))
            return torch.from_numpy(arr).unsqueeze(0)
        im = im.convert("RGB")
        arr = np.asarray(im)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t
    except Exception:
        try:
            from torchvision.io import read_image

            t = read_image(str(path))
            if t.shape[0] == 4:
                t = t[:3]
            if t.shape[0] not in (1, 3):
                raise ValueError(f"不支持的通道数: {t.shape[0]}")
            return t
        except Exception as e:
            raise ValueError(f"无法读取图像：{path}") from e


def process_path_to_nchw(path: str | Path, cfg: DataProcConfig) -> torch.Tensor:
    """
    单样本 ``(C,H,W)``。视频为 ``(T*3,H,W)``（每帧 RGB，T 为有效帧数）。
    音频为 ``(1, n_mels, target_width)``。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if cfg.modality == "webcam":
        raise ValueError("实时视频请先用摄像头采集 BGR 帧，再调用 bgr_frame_to_nchw。")
    if cfg.modality == "audio":
        return mel_spectrogram_file(path, cfg)
    if cfg.modality == "video":
        return video_clip_to_nchw(path, cfg)
    chw = load_image_chw_uint8(path)
    return chw_uint8_to_tensor_prep(
        chw,
        use_hist_equalize=bool(cfg.use_hist_equalize),
        normalize_mode=str(cfg.normalize_mode),
        target_h=int(cfg.target_height),
        target_w=int(cfg.target_width),
    )


def bgr_frame_to_nchw(bgr: np.ndarray, cfg: DataProcConfig) -> torch.Tensor:
    """OpenCV BGR ``uint8 (H,W,3)`` → 与图像支路相同的 ``(3,H,W)`` float 张量。"""
    cv2 = __import__("cv2", fromlist=["cv2"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return chw_uint8_to_tensor_prep(
        chw,
        use_hist_equalize=bool(cfg.use_hist_equalize),
        normalize_mode=str(cfg.normalize_mode),
        target_h=int(cfg.target_height),
        target_w=int(cfg.target_width),
    )


def tensor_nchw_summary(x: torch.Tensor) -> str:
    if x.dim() != 3:
        return f"期望 3 维 CHW，实际 shape={tuple(x.shape)}"
    c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    return f"CHW=({c},{h},{w}) dtype={x.dtype} min={float(x.min()):.4f} max={float(x.max()):.4f}"


def guess_modality(path: str | Path) -> str:
    """按扩展名启发式返回 ``image|audio|video``。"""
    s = Path(path).suffix.lower()
    if s in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return "video"
    if s in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        return "audio"
    return "image"
