"""图像：直方图均衡、缩放与归一化（输出 float32 CHW）。"""

from __future__ import annotations

import numpy as np


def _equalize_rgb_u8(rgb_hwc: np.ndarray) -> np.ndarray:
    """对 uint8 RGB HWC 做 Y 通道直方图均衡（亮度）。"""
    x = rgb_hwc.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    yi = y.astype(np.uint8)
    hist = np.bincount(yi.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0.0
    scale = float(yi.size - cdf_min)
    lut = np.zeros(256, dtype=np.uint8)
    if scale > 0:
        for v in range(256):
            lut[v] = np.clip(255.0 * (cdf[v] - cdf_min) / scale, 0, 255).astype(np.uint8)
    y_eq = lut[yi]
    # 简单保持色度：按比例缩放 RGB
    ratio = np.divide(y_eq.astype(np.float32), np.maximum(y.astype(np.float32), 1e-6))
    out = np.stack([r * ratio, g * ratio, b * ratio], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def chw_uint8_to_tensor_prep(
    chw_u8: torch.Tensor,
    *,
    use_hist_equalize: bool,
    normalize_mode: str,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    ``chw_u8``: uint8 ``(C,H,W)``，``C`` 为 1 或 3。
    返回 float32 ``(C,target_h,target_w)``。
    """
    if chw_u8.dtype != torch.uint8:
        chw_u8 = chw_u8.to(torch.uint8)
    c = int(chw_u8.shape[0])
    if c not in (1, 3):
        raise ValueError(f"仅支持 C=1 或 3，当前 C={c}")

    if use_hist_equalize:
        if c == 1:
            plane = chw_u8[0].numpy()
            hist = np.bincount(plane.ravel(), minlength=256).astype(np.float64)
            cdf = np.cumsum(hist)
            cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0.0
            scale = float(plane.size - cdf_min)
            lut = np.zeros(256, dtype=np.uint8)
            if scale > 0:
                for v in range(256):
                    lut[v] = np.clip(255.0 * (cdf[v] - cdf_min) / scale, 0, 255).astype(np.uint8)
            chw_u8 = torch.from_numpy(lut[plane]).unsqueeze(0).to(torch.uint8)
        else:
            hwc = chw_u8.permute(1, 2, 0).numpy()
            hwc = _equalize_rgb_u8(hwc)
            chw_u8 = torch.from_numpy(hwc).permute(2, 0, 1).contiguous()

    x = chw_u8.float()
    if normalize_mode == "none":
        pass
    elif normalize_mode == "zero_one":
        x = x / 255.0
    elif normalize_mode == "imagenet":
        if c != 3:
            raise ValueError("ImageNet 归一化需要 3 通道。")
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
        x = (x / 255.0 - mean) / std
    else:
        raise ValueError(f"未知 normalize_mode: {normalize_mode!r}")

    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(int(target_h), int(target_w)), mode="bilinear", align_corners=False)
    return x.squeeze(0)
