from __future__ import annotations

from typing import List
from pathlib import Path
from PIL import Image
import torch

from ocr.sam_postprocessing import boxes_from_results, save_crops


def load_image(image_path: Path) -> Image.Image:
    """이미지 업로드 함수"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")

def run_sam(image_path: Path, model):
    prompts = ["vegetable"]
    model.set_image(str(image_path))
    results = model(text=prompts)
    boxes_tensor = boxes_from_results(results=results)
    return save_crops(image_path=image_path, boxes_xyxy=boxes_tensor)
