from typing import List, Dict


def mock_detect_ingredients() -> List[str]:
    return ["계란", "양파", "대파", "치즈", "햄"]


def mock_get_dish_candidates(ingredients: List[str]) -> List[str]:
    # 원래는 vector_db/search 결과로 대체
    return ["계란볶음밥", "치즈오믈렛", "햄야채볶음", "파송송계란국", "양파계란덮밥"]


def mock_get_links_for_dish(dish: str) -> Dict:
    # 실제론 recipe_search/youtube_scraper 결과로 대체
    return {
        "youtube": [
            {"title": f"{dish} 초간단 레시피", "channel": "요리왕", "url": "https://www.youtube.com/"},
            {"title": f"{dish} 실패없는 비법", "channel": "집밥연구소", "url": "https://www.youtube.com/"},
            {"title": f"{dish} 10분 완성", "channel": "오늘의밥상", "url": "https://www.youtube.com/"},
        ],
        "sites": [
            {"title": f"{dish} - 만개의레시피", "url": "https://www.10000recipe.com/"},
            {"title": f"{dish} - 네이버 레시피", "url": "https://www.naver.com/"},
        ],
        "substitutes": [
            {"from": "햄", "to": ["베이컨", "참치", "두부"]},
            {"from": "치즈", "to": ["모짜렐라", "체다", "우유+버터"]},
        ],
    }

