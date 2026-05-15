"""预处理与多模态配置（可 JSON 序列化，供 UI 与工作流共用）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

Modality = Literal["image", "audio", "video", "webcam"]
NormalizeMode = Literal["none", "zero_one", "imagenet"]


@dataclass
class DataProcConfig:
    """数据管道配置：与 Input NCHW 对齐时由 UI 或调用方填写 target_*。"""

    modality: Modality = "image"

    # 目标张量空间尺寸（与画布 Input.height/width 一致；通道由数据决定或折叠为 1）
    target_height: int = 224
    target_width: int = 224

    # 图像
    use_hist_equalize: bool = False
    normalize_mode: NormalizeMode = "zero_one"

    # 音频 → 梅尔谱，再视为单通道「伪图像」(1, n_mels, target_width)
    audio_sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = None  # None = sr/2
    mel_log_floor: float = 1e-6

    # 视频：抽帧后每帧走图像预处理，沿时间维堆叠为通道 (T*C, H, W) 或仅取一帧 —— 默认堆叠通道需 Input.channels=T*3
    video_max_frames: int = 8
    video_uniform_sample: bool = True
    bad_frame_var_threshold: float = 1e-5

    # 实时视频：帧队列（满则丢旧，控制延迟）
    webcam_device_index: int = 0
    webcam_queue_max: int = 2
    webcam_width: int = 640
    webcam_height: int = 480

    # 数据集划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # 时间对齐（秒）：两侧时间戳差超过此值则不配对
    align_max_delta_sec: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataProcConfig:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
