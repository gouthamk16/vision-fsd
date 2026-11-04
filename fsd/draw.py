"""Image drawing primitives shared by every view."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
DARK_BG = (24, 24, 24)
LIGHT_BG = (244, 244, 244)


def blend(image: np.ndarray, color, alpha: np.ndarray | float) -> np.ndarray:
    """Alpha-blend `color` (BGR tuple or HxWx3 image) over `image`."""
    if isinstance(alpha, np.ndarray) and alpha.ndim == 2:
        alpha = alpha[..., None]
    color = np.asarray(color, dtype=np.float32)
    return (image.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)


def colorize(values: np.ndarray, vmin: float, vmax: float, cmap: int) -> np.ndarray:
    """Map values to BGR with an OpenCV colormap; works on 1-D or 2-D arrays."""
    if values.size == 0:  # cv2.applyColorMap returns None for empty input
        return np.empty((*values.shape, 3), dtype=np.uint8)
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8).reshape(-1, 1), cmap)
    return colors.reshape(*values.shape, 3)


def put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color, max_width: int | None = None) -> None:
    """Draw anti-aliased text, shrinking it until it fits in `max_width` pixels."""
    if max_width is not None:
        width = cv2.getTextSize(text, FONT, scale, 1)[0][0]
        if width > max_width:
            scale *= max_width / width
    cv2.putText(image, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


def with_header(image: np.ndarray, title: str, subtitle: str = "", dark: bool = True) -> np.ndarray:
    """Stack a title band above `image`, sized for the image width."""
    width = image.shape[1]
    scale = float(np.clip(width / 1400.0, 0.5, 0.9))
    line_h = int(34 * scale) + 8
    band = np.full((line_h * (2 if subtitle else 1) + 10, width, 3), DARK_BG if dark else LIGHT_BG, dtype=np.uint8)
    fg, sub = ((240, 240, 240), (190, 215, 255)) if dark else ((30, 30, 30), (120, 70, 40))
    put_text(band, title, (12, line_h), scale, fg, width - 24)
    if subtitle:
        put_text(band, subtitle, (12, 2 * line_h), scale * 0.8, sub, width - 24)
    return np.vstack([band, image])


def label_band(image: np.ndarray, text: str) -> None:
    """Overlay a one-line label strip across the top of `image`, in place."""
    h = max(22, image.shape[0] // 14)
    image[:h] = (image[:h] * 0.35).astype(np.uint8)
    put_text(image, text, (8, int(h * 0.72)), h / 40.0, (255, 255, 255), image.shape[1] - 16)


def splat(image: np.ndarray, px: np.ndarray, colors: np.ndarray, radius: int = 1) -> None:
    """Paint each (col, row) pixel as a filled disc, in order, so later points win."""
    h, w = image.shape[:2]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            rows, cols = px[:, 1] + dr, px[:, 0] + dc
            ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            image[rows[ok], cols[ok]] = colors[ok]
