from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image
import torch


def boxes_from_results(results) -> torch.Tensor:
    if isinstance(results, (list, tuple)):
        if not results:
            return torch.empty((0, 4))
        result = results[0]
    else:
        result = results

    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        return boxes.xyxy

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return torch.empty((0, 4))

    data = masks.data
    if data.ndim == 2:
        data = data.unsqueeze(0)

    coords = []
    for mask in data:
        ys, xs = torch.where(mask > 0)
        if ys.numel() == 0:
            continue
        x_min = xs.min().item()
        y_min = ys.min().item()
        x_max = xs.max().item()
        y_max = ys.max().item()
        coords.append([x_min, y_min, x_max, y_max])

    if not coords:
        return torch.empty((0, 4))
    return torch.tensor(coords, device=data.device, dtype=torch.float32)


def save_crops(
    image_path: Path,
    boxes_xyxy: torch.Tensor,
) -> List:
    image = Image.open(image_path).convert("RGB")

    saved_img: List = []
    for idx, box in enumerate(boxes_xyxy):
        left, top, right, bottom = [int(x) for x in box.tolist()]
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        saved_img.append(crop)

    return saved_img
