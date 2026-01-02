"""
OCR 파이프라인 메인 모듈
이미지에서 식자재를 추출하는 통합 파이프라인
예시.py 내 함수 재구성
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from paddleocr import PaddleOCR

from modules.ocr.ocr_inference import run_ocr_with_rotations
from modules.ocr.postprocessing import clean_ocr_with_llm
from modules.ocr.yolo_inference import load_image, run_yolo


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    """리스트 아이템 중복제거 함수"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def run_ocr_pipeline(
    image_path: Path,
    rotations: Iterable[int] = (0, 90, 180, 270),
    model_name: str = "Qwen/Qwen3-1.7B",
    yolo_weights: Path | None = None,
) -> List[str]:
    """
    이미지 경로를 입력받아 최종 식자재 LIST를 반환

    Args:
        image_path: 입력 이미지 경로
        rotations: OCR을 시도할 회전 각도들 (기본: 0, 90, 180, 270도)
        model_name: LLM 후처리에 사용할 모델 이름 (기본: Qwen/Qwen3-1.7B)
        yolo_weights: YOLO 모델 가중치 파일 경로 (선택사항)

    Returns:
        식자재 이름 리스트
    """
    ocr = PaddleOCR(lang="korean", use_angle_cls=True)
    raw_texts = run_ocr_with_rotations(ocr, image_path, rotations=rotations)
    llm_items = clean_ocr_with_llm(raw_texts, model_name=model_name)

    if yolo_weights is None:
        return llm_items

    image = load_image(image_path)
    yolo_items = run_yolo(image, str(yolo_weights))
    return _dedupe_keep_order([*yolo_items, *llm_items])
