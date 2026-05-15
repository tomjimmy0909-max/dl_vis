"""视频：抽帧、坏帧过滤、按图像管线处理并沿通道堆叠多帧。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dl_vis.logic.dataproc.config import DataProcConfig
from dl_vis.logic.dataproc.image_prep import chw_uint8_to_tensor_prep


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise ImportError("视频处理需要 OpenCV：pip install opencv-python-headless") from e
    return cv2


def iter_video_frames(path: str | Path, *, max_frames: int, uniform: bool) -> list[np.ndarray]:
    """返回 BGR ``uint8 (H,W,3)`` 帧列表。"""
    cv2 = _import_cv2()
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs: list[int]
    if total > 0 and uniform and max_frames > 0:
        idxs = np.linspace(0, max(0, total - 1), num=min(max_frames, total), dtype=np.int64).tolist()
    else:
        idxs = []
    frames: list[np.ndarray] = []
    if idxs:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    else:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    cap.release()
    return frames


def _frame_variance_bgr(bgr: np.ndarray) -> float:
    gray = np.asarray(0.299 * bgr[..., 2] + 0.587 * bgr[..., 1] + 0.114 * bgr[..., 0], dtype=np.float64)
    return float(gray.var())


def video_clip_to_nchw(path: str | Path, cfg: DataProcConfig) -> torch.Tensor:
    """
    抽帧后经 ``chw_uint8_to_tensor_prep``，将各帧 ``(C,H,W)`` 在通道维拼接为
    ``(T*C, target_h, target_w)``，以便单独一层 Conv 可将 T*C 视作输入通道。
    """
    frames_bgr = iter_video_frames(
        path, max_frames=int(cfg.video_max_frames), uniform=bool(cfg.video_uniform_sample)
    )
    if not frames_bgr:
        raise ValueError("视频未读取到任何帧。")

    cv2 = _import_cv2()
    pieces: list[torch.Tensor] = []
    for bgr in frames_bgr:
        if _frame_variance_bgr(bgr) < float(cfg.bad_frame_var_threshold):
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        if chw.shape[0] != 3:
            raise ValueError("仅支持彩色视频帧转 RGB。")
        t = chw_uint8_to_tensor_prep(
            chw,
            use_hist_equalize=bool(cfg.use_hist_equalize),
            normalize_mode=str(cfg.normalize_mode),
            target_h=int(cfg.target_height),
            target_w=int(cfg.target_width),
        )
        pieces.append(t)
    if not pieces:
        raise ValueError("过滤坏帧后无有效帧，请调低 bad_frame_var_threshold 或检查视频。")
    return torch.cat(pieces, dim=0)
