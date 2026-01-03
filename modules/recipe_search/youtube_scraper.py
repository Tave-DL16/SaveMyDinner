"""
이광호
 # 요리명 → 유튜브 검색 → 스크립트
modules.recipe_search.youtube_scraper의 Docstring

Args:
        dish_name: 요리명 (예: "김치찌개", "된장국")
        num_results: 반환할 레시피 수 (기본값: 3)
    
Returns:
    {
        "1": {"dish_name": str, "video_url": str, "video_title": str, "ingredients": list, "steps": list},
        "2": {...},
        "3": {...}
    }
"""

import os
import re
import subprocess
import json
from typing import Optional
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

load_dotenv()   # gemini api key


def search_youtube(dish_name: str, max_results: int = 5) -> list[dict]:
    """
    YouTube에서 요리 영상 검색 (yt-dlp 사용)
    
    Args:
        dish_name: 검색할 요리명
        max_results: 반환할 최대 영상 수 (기본값: 5)
    
    Returns:
        영상 정보 리스트 [{video_id, title, url}, ...]
    """
    search_query = f"ytsearch{max_results}:{dish_name} 레시피"
    
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--no-warnings",
                search_query
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                video_data = json.loads(line)
                videos.append({
                    "video_id": video_data.get("id", ""),
                    "title": video_data.get("title", ""),
                    "url": video_data.get("url", f"https://www.youtube.com/watch?v={video_data.get('id', '')}")
                })
            except json.JSONDecodeError:
                continue
        
        return videos
        
    except subprocess.TimeoutExpired:
        print("YouTube 검색 시간 초과")
        return []
    except FileNotFoundError:
        print("yt-dlp 설치 필요")
        return []
    except Exception as e:
        print(f"YouTube 검색 오류: {e}")
        return []


def extract_subtitles(video_id: str) -> Optional[str]:
    """
    YouTube 영상 자막 추출
    우선순위: 한국어 자막 → 영어 자막
    
    Args:
        video_id: YouTube 영상 ID
    
    Returns:
        자막 텍스트 (없으면 None)
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        
        # 1순위: 한국어 자막 시도
        try:
            transcript = ytt_api.fetch(video_id, languages=['ko'])
            return " ".join([entry.text for entry in transcript])
        except Exception:
            pass
        
        # 2순위: 영어 자막 시도
        try:
            transcript = ytt_api.fetch(video_id, languages=['en'])
            return " ".join([entry.text for entry in transcript])
        except Exception:
            pass
        
        # 3순위: 사용 가능한 아무 자막이나 시도
        try:
            transcript = ytt_api.fetch(video_id)
            return " ".join([entry.text for entry in transcript])
        except Exception:
            pass
        
        return None
        
    except Exception as e:
        print(f"자막 추출 오류: {e}")
        return None


def parse_recipe_with_gemini(dish_name: str, subtitle_text: str) -> dict:
    """
    자막에서 레시피 추출
    
    Args:
        dish_name: 요리명
        subtitle_text: 자막 텍스트
    
    Returns:
        {"ingredients": [...], "steps": [...]}
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수 설정 필요")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""다음은 "{dish_name}" 요리 영상의 자막입니다. 이 자막을 분석하여 레시피를 추출해주세요.

    자막:
    {subtitle_text}

    다음 형식으로 정확하게 응답해주세요:

    [재료]
    - 재료1 (분량)
    - 재료2 (분량)
    ...

    [조리 단계]
    1. 첫 번째 단계를 지시형으로 작성하세요.
    2. 두 번째 단계를 지시형으로 작성하세요.
    ...

    주의사항:
    - 조리 단계는 반드시 지시형(~하세요, ~해주세요)으로 작성하세요.
    - 자막에서 명확하게 언급된 재료와 단계만 포함하세요.
    - 분량이 언급되지 않은 재료는 "적당량"으로 표기하세요.
    """

    response = model.generate_content(prompt)
    response_text = response.text
    
    ingredients = []
    steps = []
    
    current_section = None
    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        
        if "[재료]" in line or ("재료" in line and len(line) < 10):
            current_section = "ingredients"
            continue
        elif "[조리 단계]" in line or "조리 단계" in line or "조리단계" in line:
            current_section = "steps"
            continue
        
        if current_section == "ingredients":
            if line.startswith("-") or line.startswith("•"):
                ingredients.append(line.lstrip("-•").strip())
            elif line and not line.startswith("["):
                ingredients.append(line)
        elif current_section == "steps":
            if line and (line[0].isdigit() or line.startswith("-")):
                step_text = line.lstrip("0123456789.-) ").strip()
                if step_text:
                    steps.append(f"{len(steps) + 1}. {step_text}")
    
    return {
        "ingredients": ingredients,
        "steps": steps
    }


def get_recipe_from_youtube(dish_name: str, num_results: int = 3) -> dict:
    """
    요리명을 입력받아 YouTube에서 레시피 검색, 추출 (최대 3개 영상)
    
    Args:
        dish_name: 요리명 (예: "김치찌개", "된장국")
        num_results: 반환할 레시피 수 (기본값: 3)
    
    Returns:
        {
            "1": {"dish_name": str, "video_url": str, "video_title": str, "ingredients": list, "steps": list},
            "2": {...},
            "3": {...}
        }
    
    Raises:
        ValueError: 적절한 영상을 찾지 못한 경우
    """
    # 1. YouTube 검색
    videos = search_youtube(dish_name, max_results=10)
    
    if not videos:
        raise ValueError(f"'{dish_name}'에 대한 YouTube 영상을 찾을 수 없습니다.")
    
    # 2. 자막이 있는 영상 찾기 (최대 num_results개)
    results = {}
    count = 0
    
    for video in videos:
        if count >= num_results:
            break
            
        subtitle_text = extract_subtitles(video["video_id"])
        if subtitle_text:
            # 3. Gemini로 레시피 파싱
            recipe_data = parse_recipe_with_gemini(dish_name, subtitle_text)
            
            # 4. 결과 구성
            results[str(count + 1)] = {
                "dish_name": dish_name,
                "video_url": video["url"],
                "video_title": video["title"],
                "ingredients": recipe_data["ingredients"],
                "steps": recipe_data["steps"]
            }
            count += 1
    
    if not results:
        raise ValueError(f"'{dish_name}' 관련 영상 중 자막이 있는 영상을 찾을 수 없습니다.")
    
    return results


# 테스트용 코드
if __name__ == "__main__":
    test_dish = "두바이쫀득쿠키"
    print(f"'{test_dish}' 레시피 검색 중...")
    
    try:
        result = get_recipe_from_youtube(test_dish)
        print(f"\n---------- 검색 결과 ({len(result)}개 영상) ----------")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"오류 발생: {e}")
