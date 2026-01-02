"""
modules.ocr.ocr_inference의 Docstring
# 사진 → OCR → 텍스트
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def run_ocr_with_rotations(
    ocr: PaddleOCR,
    image_path: Path,
    rotations: Iterable[int] = (0, 90, 180, 270),
) -> List[str]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    texts: List[str] = []
    for angle in rotations:
        target = img if angle == 0 else rotate_image(img, angle)
        results = ocr.predict(target)
        if not results:
            continue
        rec_texts = results[0].get("rec_texts")
        if rec_texts:
            texts.extend(rec_texts)
    return texts

def rotate_image(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate while keeping the full canvas."""
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    rad = np.deg2rad(angle)
    new_w = int(abs(h * np.sin(rad)) + abs(w * np.cos(rad)))
    new_h = int(abs(h * np.cos(rad)) + abs(w * np.sin(rad)))
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    m[0, 2] += new_w / 2 - center[0]
    m[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(
        image_bgr,
        m,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )