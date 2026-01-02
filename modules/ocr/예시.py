#예시 파일 입니다
# (혜교) - 통합 파이프라인 구축 과정에서 더이상 사용되지 않음. 삭제 가능한 파일

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

from paddleocr import PaddleOCR
from ultralytics import YOLO

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
    """이미지 경로를 입력받아 최종 식자재 LIST 를 반환"""
    ocr = PaddleOCR(lang="korean", use_angle_cls=True)
    raw_texts = run_ocr_with_rotations(ocr, image_path, rotations=rotations)
    llm_items = clean_ocr_with_llm(raw_texts, model_name=model_name)

    if yolo_weights is None:
        return llm_items

    image = load_image(image_path)
    yolo_items = run_yolo(image, str(yolo_weights))
    return _dedupe_keep_order([*yolo_items, *llm_items])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR + LLM postprocessing pipeline")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="LLM model name for postprocessing",
    )
    parser.add_argument(
        "--rotations",
        nargs="*",
        type=int,
        default=[0, 90, 180, 270],
        help="Rotation angles to try",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="YOLO weights path to merge detections with OCR results",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_ocr_pipeline(
        image_path=args.image,
        rotations=args.rotations,
        model_name=args.model,
        yolo_weights=args.yolo_weights,
    )
    print(result)


if __name__ == "__main__":
    main()
