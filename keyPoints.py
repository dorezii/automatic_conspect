from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def detect_slide_keypoints(frame_bgr: np.ndarray) -> dict:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 0.1 * w * h:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2)

    if best is None:
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        best = np.array(
            [
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y],
            ],
            dtype=np.int32,
        )

    x, y, bw, bh = cv2.boundingRect(best.astype(np.int32))
    return {
        "points": best.astype(int).tolist(),
        "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
    }


def detect_keypoints_for_images(image_paths: Iterable[Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result[str(image_path)] = detect_slide_keypoints(frame)
    return result
