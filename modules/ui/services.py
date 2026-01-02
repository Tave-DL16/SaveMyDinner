"""
[통합 가이드]
백엔드 기능 구현이 완료되면, 아래의 import 문을 실제 모듈 경로로 수정해주세요.

(변경 예시)
from modules.ocr.main import run_ocr_pipeline as detect_ingredients
from modules.recipe_search.main import get_dish_candidates
from modules.recipe_search.main import get_recipe_links

현재는 'mock_services.py'를 연결하여 테스트 중입니다.
"""

import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

from dotenv import load_dotenv
load_dotenv()

# 아래 import를 나중에 바꾸게 됩니다
from .mock_services import (
    mock_detect_ingredients,
    mock_get_dish_candidates,
    mock_get_links_for_dish,
)

_ocr_pipeline = None

def _get_ocr_pipeline():
    """OCR 파이프라인을 lazy load하는 함수"""
    global _ocr_pipeline
    if _ocr_pipeline is None:
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from modules.ocr.main import run_ocr_pipeline
            _ocr_pipeline = run_ocr_pipeline
        except ImportError as e:
            print(f"OCR 모듈 import 실패: {e}")
            print("Mock 데이터를 사용합니다.")
            return None
    return _ocr_pipeline


def detect_ingredients(image_file: Any) -> List[str]:
    """
    이미지 파일(업로드 파일)을 받아 재료 리스트를 반환.
    Streamlit 업로드 파일을 임시 파일로 저장 후 OCR 파이프라인 실행.
    """
    if image_file is None:
        return []
    
    # OCR 모듈 사용
    run_ocr_pipeline = _get_ocr_pipeline()

    # OCR 모듈을 사용할 수 없으면 mock 데이터 반환
    if run_ocr_pipeline is None:
        print("OCR 모듈을 사용할 수 없어 Mock 데이터를 사용합니다.")
        return mock_detect_ingredients()

    # Streamlit 업로드 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        # OCR 파이프라인 실행
        yolo_weights = Path(__file__).parent.parent / "ocr" / "best.pt"

        ingredients = run_ocr_pipeline(
            image_path=tmp_path,
            rotations=(0, 90, 180, 270),
            model_name="Qwen/Qwen3-1.7B",
            yolo_weights=yolo_weights if yolo_weights.exists() else None,
        )
        return ingredients
    except Exception as e:
        print(f"OCR Error: {e}")
        print("오류 발생으로 Mock 데이터를 사용합니다.")
        return mock_detect_ingredients()
    finally:
        # 임시 파일 삭제
        if tmp_path.exists():
            tmp_path.unlink()


def get_dish_candidates(ingredients: List[str]) -> List[str]:
    """
    재료 리스트를 받아 요리 후보 5개 정도 반환.
    나중에 modules/vector_db/... 로 교체 예정.
    """
    return mock_get_dish_candidates(ingredients)


def get_recipe_links(dish: str) -> Dict:
    """
    요리명을 받아 유튜브/사이트 링크 반환.
    나중에 modules/recipe_search/... 로 교체 예정.
    """
    return mock_get_links_for_dish(dish)

