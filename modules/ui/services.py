"""
[통합 가이드]
백엔드 기능 구현이 완료되면, 아래의 import 문을 실제 모듈 경로로 수정해주세요.

(변경 예시)
from modules.ocr.main import run_ocr_pipeline as detect_ingredients
from modules.recipe_search.main import get_dish_candidates
from modules.recipe_search.main import get_recipe_links

현재는 'mock_services.py'를 연결하여 테스트 중입니다.
"""

from typing import List, Dict, Any

# 아래 import를 나중에 바꾸게 됩니다
from .mock_services import (
    mock_detect_ingredients,
    mock_get_dish_candidates,
    mock_get_links_for_dish,
)


def detect_ingredients(image_file: Any) -> List[str]:
    """
    이미지 파일(업로드 파일)을 받아 재료 리스트를 반환.
    나중에 modules/ocr/... 로 교체 예정.
    """
    return mock_detect_ingredients()


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

