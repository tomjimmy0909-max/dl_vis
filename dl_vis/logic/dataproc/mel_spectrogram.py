"""音频：波形读取与梅尔频谱（纯 torch + numpy，输出 ``(1, n_mels, W)``）。"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_weight_matrix(
    n_fft: int,
    n_mels: int,
    sr: float,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    """形状 ``(n_mels, n_fft//2+1)`` 的三角梅尔滤波器组。"""
    n_freqs = n_fft // 2 + 1
    mel_lo = _hz_to_mel(np.array([f_min], dtype=np.float64))[0]
    mel_hi = _hz_to_mel(np.array([f_max], dtype=np.float64))[0]
    m_pts = np.linspace(mel_lo, mel_hi, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(np.int64)
    bins = np.clip(bins, 0, n_freqs)

    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            weights[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            weights[i, j] = (right - j) / max(right - center, 1)
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-8)
    weights /= row_sum
    return torch.from_numpy(weights)


def load_waveform_mono(path: str | Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """
    读取单声道 float32 波形 ``[-1,1]``，形状 ``(T,)``。
    当前内置：16-bit PCM WAV（多通道则取平均）；其它格式需安装 torchaudio 后扩展。
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        if sw != 2:
            raise ValueError(f"暂仅支持 16-bit PCM WAV，当前 sampwidth={sw}")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        w = torch.from_numpy(x)
        cur_sr = int(sr)
    else:
        try:
            import torchaudio  # type: ignore
        except ImportError as e:
            raise ImportError("非 WAV 音频请安装 torchaudio：pip install torchaudio") from e
        w, cur_sr = torchaudio.load(str(path))
        if w.dim() == 2 and w.size(0) > 1:
            w = w.mean(dim=0, keepdim=True)
        w = w.squeeze(0)
        cur_sr = int(cur_sr)

    if cur_sr != int(target_sr):
        try:
            import torchaudio  # type: ignore

            w = torchaudio.functional.resample(w, cur_sr, int(target_sr))
            cur_sr = int(target_sr)
        except ImportError:
            # 线性重采样（简易）
            tgt_len = int(round(w.numel() * int(target_sr) / cur_sr))
            w = F.interpolate(w.view(1, 1, -1), size=tgt_len, mode="linear", align_corners=False).view(-1)
            cur_sr = int(target_sr)
    return w, cur_sr


def waveform_to_mel_nchw(
    waveform: torch.Tensor,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float | None,
    target_height: int,
    target_width: int,
    log_floor: float,
) -> torch.Tensor:
    """``(1, target_height, target_width)``，其中 ``target_height`` 应对齐 ``n_mels``（会插值）。"""
    if f_max is None:
        f_max = float(sr) / 2.0
    wf = waveform.float()
    if wf.dim() != 1:
        wf = wf.reshape(-1)
    window = torch.hann_window(int(n_fft), device=wf.device)
    stft = torch.stft(
        wf,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        window=window,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    power = stft.abs().clamp_min(1e-10).pow(2.0)
    mel_w = _mel_weight_matrix(int(n_fft), int(n_mels), float(sr), float(f_min), float(f_max)).to(
        device=power.device, dtype=power.dtype
    )
    mel = torch.matmul(mel_w, power)  # (n_mels, T)
    mel = torch.log(mel + float(log_floor))
    mel = mel.unsqueeze(0).unsqueeze(0)  # 1,1,M,T
    mel = F.interpolate(
        mel,
        size=(int(target_height), int(target_width)),
        mode="bilinear",
        align_corners=False,
    )
    return mel.squeeze(0)  # (1,H,W)


def mel_spectrogram_file(path: str | Path, cfg) -> torch.Tensor:
    """使用 ``DataProcConfig`` 中音频字段。"""
    w, _sr = load_waveform_mono(path, int(cfg.audio_sample_rate))
    f_max = cfg.f_max
    return waveform_to_mel_nchw(
        w,
        int(cfg.audio_sample_rate),
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop_length),
        n_mels=int(cfg.n_mels),
        f_min=float(cfg.f_min),
        f_max=float(f_max) if f_max is not None else None,
        target_height=int(cfg.n_mels),
        target_width=int(cfg.target_width),
        log_floor=float(cfg.mel_log_floor),
    )
