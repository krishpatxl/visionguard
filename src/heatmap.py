from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class Heatmap:
    """
    Accumulates centroids into a density map and can save an overlay image.
    """

    def __init__(self, width: int, height: int):
        self.w = int(width)
        self.h = int(height)
        self.acc = np.zeros((self.h, self.w), dtype=np.float32)

    def add(self, x: float, y: float, weight: float = 1.0) -> None:
        xi = int(np.clip(x, 0, self.w - 1))
        yi = int(np.clip(y, 0, self.h - 1))
        self.acc[yi, xi] += float(weight)

    # Backwards-compatible alias (in case older code calls add_point)
    def add_point(self, x: float, y: float, weight: float = 1.0) -> None:
        self.add(x, y, weight=weight)

    def _render_color(self) -> np.ndarray:
        mat = self.acc.copy()
        if mat.max() > 0:
            mat = mat / mat.max()
        mat = (mat * 255).astype(np.uint8)

        # smooth so it looks nice
        mat = cv2.GaussianBlur(mat, (0, 0), sigmaX=15, sigmaY=15)
        return cv2.applyColorMap(mat, cv2.COLORMAP_JET)

    def save_overlay(self, frame_bgr: np.ndarray, out_path: str | Path, alpha: float = 0.45) -> None:
        heat = self._render_color()
        heat = cv2.resize(heat, (frame_bgr.shape[1], frame_bgr.shape[0]))

        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heat, alpha, 0)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)