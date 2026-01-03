import json
from typing import List, Sequence
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# 영어 -> 한국어 식자재
INGREDIENT_TRANSLATION = {
    'chilli': '고추',
    'greenonion': '대파',
    'garlic': '마늘',
    'carrot': '당근',
    'Onion_peel': '양파',
    'Onion_bag': '양파',
    'onion': '양파',
    'radish': '무',
    'Radish': '무',
    'potato': '감자',
    'green_onion': '대파',
    'green onion': '대파',
}


def load_image(image_path: Path) -> Image.Image:
    """이미지 업로드 함수"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def check_conf(data):
    """신뢰도 0.6 이상의 식자재만 인식하는 함수 + 한글 변환"""
    item_list = []
    for _ , values in data.items():
        if not values:
            # print(f"{key}: no detections")
            continue

        for value in values:
            conf = value.get("confidence") if isinstance(value, dict) else None
            mask = value.get("class_name")
            if conf >= 0.6:
                # 영어 -> 한글 변환
                korean_name = INGREDIENT_TRANSLATION.get(mask, mask)
                item_list.append(korean_name)
    return item_list


def split_into_quadrants(image: Image.Image) -> Sequence[Image.Image]:
    """이미지를 4등분하는 함수"""
    width, height = image.size
    mid_x = width // 2
    mid_y = height // 2
    boxes = [
        (0,0,width,height),
        (0, 0, mid_x, mid_y),
        (mid_x, 0, width, mid_y),
        (0, mid_y, mid_x, height),
        (mid_x, mid_y, width, height),
    ]
    return [image.crop(box) for box in boxes]

def serialize_boxes(result) -> List[dict]:
    """Convert YOLO result boxes to JSON-serializable dictionaries."""
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.tolist()
    confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(xyxy)
    cls_ids = boxes.cls.tolist() if boxes.cls is not None else [None] * len(xyxy)

    detections = []
    for coords, conf, cls_id in zip(xyxy, confs, cls_ids):
        cls_name = names.get(int(cls_id), str(int(cls_id))) if cls_id is not None else None
        detections.append(
            {
                "bbox_xyxy": coords,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
            }
        )
    return detections

def run_yolo(image , weights):
    model = YOLO(weights)   
    result_dict = {}

    for idx, img in enumerate(split_into_quadrants(image)):
        result = list(model(img))[0]
        result_dict[idx] = serialize_boxes(result)

    return check_conf(result_dict)
