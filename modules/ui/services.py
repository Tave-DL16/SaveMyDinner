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

# OCR 파이프라인 및 리소스
_ocr_pipeline = None
_ocr_engine = None
_sam_model = None

# VectorDB 검색기
_recipe_searcher = None

# YouTube 레시피 검색 함수
_youtube_recipe_func = None

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


def _get_ocr_resources():
    """OCR 엔진과 SAM 모델을 lazy load하는 함수"""
    global _ocr_engine, _sam_model
    if _ocr_engine is None or _sam_model is None:
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from modules.ocr.main import OCR_ENGINE, SAM_MODEL
            _ocr_engine = OCR_ENGINE
            _sam_model = SAM_MODEL
        except ImportError as e:
            print(f"OCR 리소스 import 실패: {e}")
            print("Mock 데이터를 사용합니다.")
            return None, None
    return _ocr_engine, _sam_model


def _get_recipe_searcher():
    """VectorDB RecipeSearcher를 lazy load하는 함수"""
    global _recipe_searcher
    if _recipe_searcher is None:
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from modules.vector_db.search import RecipeSearcher
            _recipe_searcher = RecipeSearcher()
            print("✅ VectorDB RecipeSearcher 로드 완료")
        except Exception as e:
            print(f"VectorDB 모듈 import 실패: {e}")
            print("Mock 데이터를 사용합니다.")
            return None
    return _recipe_searcher


def _get_youtube_recipe_func():
    """YouTube 레시피 검색 함수를 lazy load하는 함수"""
    global _youtube_recipe_func
    if _youtube_recipe_func is None:
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from modules.recipe_search.youtube_scraper import get_recipe_from_youtube
            _youtube_recipe_func = get_recipe_from_youtube
            print("✅ YouTube 레시피 검색 모듈 로드 완료")
        except Exception as e:
            print(f"YouTube 레시피 검색 모듈 import 실패: {e}")
            print("Mock 데이터를 사용합니다.")
            return None
    return _youtube_recipe_func


def detect_ingredients(image_file: Any) -> List[str]:
    """
    이미지 파일(업로드 파일)을 받아 재료 리스트를 반환.
    Streamlit 업로드 파일을 임시 파일로 저장 후 OCR 파이프라인 실행.
    """
    if image_file is None:
        return []
    
    # OCR 모듈 사용
    run_ocr_pipeline = _get_ocr_pipeline()
    ocr_engine, sam_model = _get_ocr_resources()

    # OCR 모듈을 사용할 수 없으면 mock 데이터 반환
    if run_ocr_pipeline is None or ocr_engine is None or sam_model is None:
        print("OCR 모듈을 사용할 수 없어 Mock 데이터를 사용합니다.")
        return mock_detect_ingredients()

    # Streamlit 업로드 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp_file:
        tmp_file.write(image_file.getvalue())
        tmp_path = Path(tmp_file.name)

    try:
        # OCR 파이프라인 실행
        ingredients = run_ocr_pipeline(
            ocr_engine=ocr_engine,
            sam_model=sam_model,
            image_path=tmp_path,
            rotations=(0, 90, 180, 270),
            model_name_llm="Qwen/Qwen3-1.7B",
            model_name_vlm="Qwen/Qwen3-VL-2B-Instruct",
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
    VectorDB 하이브리드 검색으로 레시피 추천.
    """
    if not ingredients:
        return []

    # VectorDB RecipeSearcher 로드
    searcher = _get_recipe_searcher()

    # VectorDB를 사용할 수 없으면 mock 데이터 반환
    if searcher is None:
        print("VectorDB를 사용할 수 없어 Mock 데이터를 사용합니다.")
        return mock_get_dish_candidates(ingredients)

    try:
        print(f"\n{'='*50}")
        print(f"🛒 재료 기반 레시피 검색: {ingredients}")
        print(f"{'='*50}")

        # 하이브리드 검색 실행 (벡터 유사도 60% + 키워드 매칭 40%)
        top_recipes = searcher.hybrid_search(ingredients, n_results=5)

        # 정제된 요리명만 추출
        recipe_names = [r['name'] for r in top_recipes]

        print(f"\n✅ 추천 레시피 ({len(recipe_names)}개):")
        for idx, recipe in enumerate(top_recipes, 1):
            print(f"   {idx}. {recipe['name']} (적합도: {recipe['score']}%)")

        print(f"{'='*50}\n")

        return recipe_names

    except Exception as e:
        print(f"VectorDB 검색 오류: {e}")
        print("오류 발생으로 Mock 데이터를 사용합니다.")
        return mock_get_dish_candidates(ingredients)


def get_recipe_links(dish: str) -> Dict:
    """
    요리명을 받아 유튜브 레시피 링크 반환.
    YouTube 자막을 분석하여 재료와 조리 단계를 포함한 레시피 정보 제공.
    """
    if not dish:
        return {"youtube": []}

    # YouTube 레시피 검색 함수 로드
    get_recipe_from_youtube = _get_youtube_recipe_func()

    # YouTube 레시피 검색을 사용할 수 없으면 mock 데이터 반환
    if get_recipe_from_youtube is None:
        print("YouTube 레시피 검색을 사용할 수 없어 Mock 데이터를 사용합니다.")
        return mock_get_links_for_dish(dish)

    try:
        print(f"\n{'='*50}")
        print(f"🎬 YouTube 레시피 검색: {dish}")
        print(f"{'='*50}")

        # YouTube에서 레시피 검색 (최대 3개)
        recipe_results = get_recipe_from_youtube(dish, num_results=3)

        # UI 형식에 맞게 변환
        youtube_list = []
        for key, recipe_data in recipe_results.items():
            youtube_list.append({
                "title": recipe_data["video_title"],
                "channel": recipe_data.get("dish_name", dish),  # 채널명 대신 요리명 사용
                "url": recipe_data["video_url"],
                # 추가 정보 (UI에서 필요 시 사용 가능)
                "ingredients": recipe_data.get("ingredients", []),
                "steps": recipe_data.get("steps", [])
            })

        print(f"\n✅ 검색 완료: {len(youtube_list)}개 영상 찾음")
        for idx, video in enumerate(youtube_list, 1):
            print(f"   {idx}. {video['title']}")
            print(f"      재료: {len(video.get('ingredients', []))}개, 단계: {len(video.get('steps', []))}개")

        print(f"{'='*50}\n")

        return {"youtube": youtube_list}

    except ValueError as e:
        # 영상을 찾지 못한 경우
        print(f"⚠️ YouTube 레시피 검색 실패: {e}")
        print("Mock 데이터를 사용합니다.")
        return mock_get_links_for_dish(dish)
    except Exception as e:
        print(f"YouTube 레시피 검색 오류: {e}")
        print("오류 발생으로 Mock 데이터를 사용합니다.")
        return mock_get_links_for_dish(dish)


# def get_recipe_links(dish: str) -> Dict:
#     """
#     요리명을 받아 유튜브/사이트 링크 반환 (Mock 버전).
#     """
#     return mock_get_links_for_dish(dish)
