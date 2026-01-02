from typing import List, Dict, Any

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

