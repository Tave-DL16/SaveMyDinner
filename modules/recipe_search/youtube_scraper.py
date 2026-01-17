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
import json
import glob
import shutil
import subprocess
from typing import Optional, List, Dict
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

load_dotenv()


# -----------------------------
# 1) YouTube search (yt-dlp)
# -----------------------------
def search_youtube(dish_name: str, max_results: int = 5) -> List[Dict]:
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
                search_query,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                video_data = json.loads(line)
                vid = video_data.get("id", "") or ""
                title = video_data.get("title", "") or ""
                url = video_data.get("url") or f"https://www.youtube.com/watch?v={vid}"

                if vid:
                    videos.append(
                        {
                            "video_id": vid,
                            "title": title,
                            "url": url,
                        }
                    )
            except json.JSONDecodeError:
                continue

        return videos

    except subprocess.TimeoutExpired:
        print("YouTube 검색 시간 초과")
        return []
    except FileNotFoundError:
        print("yt-dlp 설치 필요 (brew install yt-dlp 또는 pipx/pip 설치)")
        return []
    except Exception as e:
        print(f"YouTube 검색 오류: {e}")
        return []


# -----------------------------
# 2) Subtitles extraction
#    2-1) youtube-transcript-api (preferred)
#    2-2) yt-dlp fallback (auto-sub 포함)
# -----------------------------
def _extract_subtitles_with_transcript_api(video_id: str) -> Optional[str]:
    """
    youtube-transcript-api 로 자막 추출
    우선순위:
      1) 한국어 자막
      2) 영어 자막
      3) 번역 가능한 경우 한국어 번역 시도
    """
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

        # 1) 한국어(수동/자동)
        for lang in ["ko", "ko-KR", "ko-KP"]:
            try:
                t = transcripts.find_transcript([lang])
                data = t.fetch()
                return " ".join([x.get("text", "") for x in data]).strip() or None
            except Exception:
                pass

        # 2) 영어(수동/자동)
        for lang in ["en", "en-US", "en-GB"]:
            try:
                t = transcripts.find_transcript([lang])
                data = t.fetch()
                return " ".join([x.get("text", "") for x in data]).strip() or None
            except Exception:
                pass

        # 3) 번역 가능한 경우 (아무 자막 -> ko 번역)
        try:
            for t in transcripts:
                if getattr(t, "is_translatable", False):
                    data = t.translate("ko").fetch()
                    return " ".join([x.get("text", "") for x in data]).strip() or None
        except Exception:
            pass

        return None
    except Exception:
        return None


def _vtt_to_text(vtt_raw: str) -> str:
    """
    아주 간단한 VTT 텍스트화:
    - 타임코드/헤더 제거
    - 빈 줄 제거
    """
    lines = []
    for line in vtt_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if "-->" in line:
            continue
        # cue setting 라인 등 제거(필요 시 확장)
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def _extract_subtitles_with_ytdlp(video_url: str, tmp_dir: str = "tmp_subs") -> Optional[str]:
    """
    yt-dlp로 자막/자동자막 다운로드 후 텍스트로 변환
    - 영상 다운로드 없이 자막만
    - ko,en 우선 시도
    """
    try:
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-sub",
            "--write-auto-sub",
            "--sub-lang",
            "ko,en",
            "--sub-format",
            "vtt",
            "-o",
            f"{tmp_dir}/%(id)s.%(ext)s",
            video_url,
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)

        # 생성된 vtt 파일 찾기
        vtts = glob.glob(os.path.join(tmp_dir, "*.vtt"))
        if not vtts:
            return None

        # 가장 최신 파일을 읽어서 텍스트화
        path = sorted(vtts, key=os.path.getmtime)[-1]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        text = _vtt_to_text(raw)
        return text or None

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        print("yt-dlp 설치 필요 (brew install yt-dlp 또는 pipx/pip 설치)")
        return None
    except Exception:
        return None
    finally:
        # tmp 정리 (원하면 주석 처리)
        try:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def extract_subtitles(video_id: str, video_url: str) -> Optional[str]:
    """
    자막 추출:
      1) youtube-transcript-api 우선
      2) 실패 시 yt-dlp로 자막/자동자막 fallback
    """
    text = _extract_subtitles_with_transcript_api(video_id)
    if text:
        return text

    # fallback
    return _extract_subtitles_with_ytdlp(video_url)


# -----------------------------
# 3) OpenAI parsing
# -----------------------------
def parse_recipe_with_openai(dish_name: str, subtitle_text: str) -> dict:
    """
    자막에서 레시피 추출 (OpenAI 사용)

    Returns:
        {"ingredients": [...], "steps": [...]}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수 설정 필요")

    client = OpenAI(api_key=api_key)

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 요리 레시피를 추출하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    response_text = (response.choices[0].message.content or "").strip()

    # ---- 파싱 ----
    ingredients: List[str] = []
    steps: List[str] = []

    current_section = None
    for raw_line in response_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # 섹션 감지
        lowered = line.replace(" ", "")
        if line.startswith("[재료]") or lowered in ("재료", "[재료]"):
            current_section = "ingredients"
            continue
        if line.startswith("[조리 단계]") or line.startswith("[조리단계]") or "조리단계" in lowered or "조리단계" in line:
            current_section = "steps"
            continue

        # 내용 파싱
        if current_section == "ingredients":
            if line.startswith(("-", "•")):
                item = line.lstrip("-•").strip()
                if item:
                    ingredients.append(item)
            elif not line.startswith("["):
                ingredients.append(line)

        elif current_section == "steps":
            # "1. ~", "1) ~", "- ~" 모두 처리
            cleaned = line
            if cleaned and (cleaned[0].isdigit() or cleaned.startswith("-")):
                cleaned = cleaned.lstrip("0123456789.-) ").strip()
            if cleaned:
                steps.append(f"{len(steps) + 1}. {cleaned}")

    return {"ingredients": ingredients, "steps": steps}


# -----------------------------
# 4) Main function
# -----------------------------
def get_recipe_from_youtube(dish_name: str, num_results: int = 3) -> dict:
    """
    요리명을 입력받아 YouTube에서 레시피 검색, 추출 (최대 num_results개 영상)
    """
    videos = search_youtube(dish_name, max_results=10)
    if not videos:
        raise ValueError(f"'{dish_name}'에 대한 YouTube 영상을 찾을 수 없습니다.")

    results = {}
    count = 0

    for video in videos:
        if count >= num_results:
            break

        vid = video.get("video_id", "")
        url = video.get("url", "")
        if not vid or not url:
            continue

        subtitle_text = extract_subtitles(vid, url)
        if not subtitle_text:
            continue

        recipe_data = parse_recipe_with_openai(dish_name, subtitle_text)

        # 재료/단계가 너무 비었으면 스킵 (원하면 조건 완화)
        if not recipe_data.get("ingredients") and not recipe_data.get("steps"):
            continue

        results[str(count + 1)] = {
            "dish_name": dish_name,
            "video_url": url,
            "video_title": video.get("title", ""),
            "ingredients": recipe_data.get("ingredients", []),
            "steps": recipe_data.get("steps", []),
        }
        count += 1

    if not results:
        raise ValueError(f"'{dish_name}' 관련 영상 중 자막(또는 자동자막)을 찾을 수 없습니다.")

    return results


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    test_dish = "두바이쫀득쿠키"
    print(f"'{test_dish}' 레시피 검색 중...")

    try:
        result = get_recipe_from_youtube(test_dish)
        print(f"\n---------- 검색 결과 ({len(result)}개 영상) ----------")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"오류 발생: {e}")
