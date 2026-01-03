"""
태인
대체 식재료 추천 + 최종 레시피 출력

입력:
- recipe_result (이전 모듈 output)
- user_ingredients (사용자 보유 재료 리스트)

출력:
- 부족 재료
- 대체 식재료
- 대체 반영된 최종 레시피 텍스트

실행 전 필요:
<<<<<<< Updated upstream
환경변수 GEMINI_API_KEY 설정
=======
- (로컬모델 사용) transformers, torch 설치
- 모델 다운로드(처음 1회): Qwen/Qwen3-1.7B
>>>>>>> Stashed changes
"""

from __future__ import annotations

<<<<<<< Updated upstream
import os
=======
>>>>>>> Stashed changes
import json
import re
from typing import Dict, List, Optional

<<<<<<< Updated upstream
import google.generativeai as genai


DEFAULT_MODEL_NAME = "gemini-2.5-flash"


# JSON 파싱 유틸
=======
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 로컬 LLM 설정
# =========================
LOCAL_MODEL_NAME = "Qwen/Qwen3-1.7B"
# "auto"면 GPU 있으면 GPU, 없으면 CPU로 감 (CPU면 매우 느림)
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"

# 생성 파라미터(너무 크게 잡지 마)
MAX_NEW_TOKENS = 1200
TEMPERATURE = 0.2
TOP_P = 0.9

_tokenizer = None
_model = None


def _get_local_llm():
    """모델/토크나이저 1회 로드 후 재사용"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE_MAP
        )
        _model.eval()
    return _tokenizer, _model


# =========================
# JSON 파싱 유틸
# =========================
>>>>>>> Stashed changes
def _extract_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None

<<<<<<< Updated upstream
# Gemini 호출
def _call_gemini(prompt: str, model_name: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return resp.text

# 프롬프트 
=======

# =========================
# ✅ Gemini 호출 대신 로컬 모델 호출
# =========================
def _call_local_llm(prompt: str) -> str:
    tokenizer, model = _get_local_llm()

    messages = [{"role": "user", "content": prompt}]
    # Qwen 계열은 chat_template이 있는 경우가 많음
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    # prompt 부분 잘라내고 생성된 부분만 디코딩
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return content


# =========================
# 프롬프트
# =========================
>>>>>>> Stashed changes
def _build_prompt(recipe_result: Dict, user_ingredients: List[str]) -> str:
    dish_name = recipe_result.get("dish_name", "")
    video_title = recipe_result.get("video_title", "")
    video_url = recipe_result.get("video_url", "")
    raw_ings = recipe_result.get("ingredients", [])
    raw_steps = recipe_result.get("steps", [])

<<<<<<< Updated upstream
    # 안전하게 string화
    raw_ings_text = "\n".join([f"- {str(x)}" for x in raw_ings][:80])
    raw_steps_text = "\n".join([f"- {str(x)}" for x in raw_steps][:120])

    user_ing_text = ", ".join([str(x) for x in user_ingredients][:200])

    return f"""
# Role
=======
    raw_ings_text = "\n".join([f"- {str(x)}" for x in raw_ings][:80])
    raw_steps_text = "\n".join([f"- {str(x)}" for x in raw_steps][:120])
    user_ing_text = ", ".join([str(x) for x in user_ingredients][:200])

    return f"""
  # Role
>>>>>>> Stashed changes
너는 요리 편집 전문가 + 데이터 정제 AI다.

# Goal
유튜브 자막에서 추출된 투박한 레시피를 "사람이 바로 따라할 수 있게" 정리한다.
또한 사용자가 가진 재료(user_ingredients)와 비교해 부족한 재료를 찾고, 대체재를 추천한다.

# Inputs
dish_name: {dish_name}
video_title: {video_title}
<<<<<<< Updated upstream
video_url: {video_url}
=======

>>>>>>> Stashed changes

user_ingredients (사용자 보유 재료, 이 리스트 밖의 주재료는 추가 금지):
{user_ing_text}

<<<<<<< Updated upstream
raw_ingredients (자막에서 뽑혀서 분량/잡텍스트가 섞일 수 있음):
{raw_ings_text}
=======
>>>>>>> Stashed changes

raw_steps (자막에서 뽑혀서 중복/구어체/누락이 있을 수 있음):
{raw_steps_text}

# Hard Rules (위반하면 실패)
1) 새로운 "주재료"를 창작/추가하지 마라.
   - user_ingredients에 없고 raw_ingredients에도 없는 재료는 절대 추가 금지.
2) 기본 조미료(물/소금/설탕/식용유/후추)는 언급할 수 있으나, 재료 목록에 굳이 추가하지 않아도 된다.
3) 출력은 반드시 JSON 오브젝트 1개만. 다른 텍스트 절대 금지.
4) 요리명은 바꾸지 마라.

# Tasks
A) 재료 정리/정규화:
- raw_ingredients를 보고 "재료명만" 깔끔하게 정리한다.
- 분량, 단위, 괄호, 브랜드, 수식어 제거.
- 예) "닭다리살(600g)" -> "닭다리살", "양파 1/2개" -> "양파"

B) 부족 재료 및 대체재:
- 정리된 재료(clean_ingredients) 중 user_ingredients에 없는 것을 missing_ingredients로 만든다.
- missing_ingredients 각각에 대해 substitutes를 작성한다.
- 대체재는 "현실적으로 같은 역할"을 하는 재료만.
- 대체 불가면 "대체 불가"로.

C) 최종 레시피 편집:
- clean_ingredients를 기반으로 최종 재료 리스트를 만든다.
  - 대체재가 있는 경우 "대체재 (대체: 원재료)" 형태로 표기.
  - 대체 불가 재료는 그대로 남기되, tips에 "구매 권장" 같은 한 줄을 넣어도 됨.
- raw_steps를 바탕으로 조리 단계를 5~10개 이내로 정리한다.
  - 지시형(하세요/해주세요)
  - 중복 제거
  - 너무 길면 쪼개기
  - 불확실한 내용은 추측하지 말고 생략
<<<<<<< Updated upstream
=======
  - STEP은 반드시 1,2,3,4 리스트 형태로 출력
>>>>>>> Stashed changes

# Output JSON Schema
{{
  "dish_name": "{dish_name}",
<<<<<<< Updated upstream
  "video_url": "{video_url}",
  "video_title": "{video_title}",
  "clean_ingredients": ["재료1","재료2"],
  "missing_ingredients": ["부족재료1"],
  "substitutes": {{"부족재료1":"대체재 or 대체 불가"}},
  "final_recipe": {{
    "title": "{dish_name}",
    "ingredients": ["정리된재료/대체반영재료"],
    "steps": ["1. ...", "2. ..."],
    "tips": ["팁1", "팁2"]
=======
  "video_title": "{video_title}",
  "clean_ingredients": ["재료1","재료2"],
  "missing_ingredients": ["부족재료1"],
  "final_recipe": {{
    "title": "{dish_name}",
    "steps": ["1. ...", "2. ...","3. ....","4. ..."],
>>>>>>> Stashed changes
  }}
}}
""".strip()


<<<<<<< Updated upstream
# 사람이 읽기 좋은 텍스트로 변환
=======
# =========================
# 사람이 읽기 좋은 텍스트로 변환
# =========================
>>>>>>> Stashed changes
def _render_final_text(data: Dict) -> str:
    dish = data.get("dish_name", "")
    vt = data.get("video_title", "")
    url = data.get("video_url", "")

    final = data.get("final_recipe", {}) or {}
    ings = final.get("ingredients", []) or []
    steps = final.get("steps", []) or []
    tips = final.get("tips", []) or []

    subs = data.get("substitutes", {}) or {}
    missing = data.get("missing_ingredients", []) or []

    lines: List[str] = []
    lines.append(f"{dish}".strip())
    if vt:
        lines.append(f"{vt}")
    if url:
        lines.append(f"{url}")

    lines.append("")
    lines.append("재료")
    for x in ings:
        lines.append(f"- {x}")

    lines.append("")
    lines.append("조리 방법")
    for s in steps:
        lines.append(f"{s}" if re.match(r"^\d+\.", str(s).strip()) else f"- {s}")

    lines.append("")
    lines.append("대체 식재료")
    if missing and subs:
        for m in missing:
            v = subs.get(m, "대체 불가")
            lines.append(f"- {m} → {v}")
    else:
        lines.append("- (없음)")

    if tips:
        lines.append("")
        lines.append("팁")
        for t in tips:
            lines.append(f"- {t}")

    return "\n".join(lines).strip()


<<<<<<< Updated upstream
# 통합에서 호출할 함수 
def format_recipe_with_substitutes(
    recipe_result: Dict,
    user_ingredients: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict:
    prompt = _build_prompt(recipe_result, user_ingredients)
    raw = _call_gemini(prompt, model_name=model_name)

    parsed = _extract_json_object(raw)
    if not parsed:
        # 실패 시 최소 결과라도 리턴 (통합이 죽지 않게)
=======
# =========================
# 통합에서 호출할 함수(외부 공개)
# =========================
def format_recipe_with_substitutes(
    recipe_result: Dict,
    user_ingredients: List[str],
) -> Dict:
    prompt = _build_prompt(recipe_result, user_ingredients)

    raw = _call_local_llm(prompt)  # 여기만 바뀜 (API → 로컬)

    parsed = _extract_json_object(raw)
    if not parsed:
>>>>>>> Stashed changes
        return {
            "dish_name": recipe_result.get("dish_name", ""),
            "video_url": recipe_result.get("video_url", ""),
            "video_title": recipe_result.get("video_title", ""),
            "clean_ingredients": [],
            "missing_ingredients": [],
            "substitutes": {},
            "final_recipe": {
                "title": recipe_result.get("dish_name", ""),
                "ingredients": recipe_result.get("ingredients", []) or [],
                "steps": recipe_result.get("steps", []) or [],
                "tips": ["LLM 출력 파싱 실패 - 원본 레시피를 그대로 사용하세요."]
            },
            "final_text": _render_final_text({
                "dish_name": recipe_result.get("dish_name", ""),
                "video_url": recipe_result.get("video_url", ""),
                "video_title": recipe_result.get("video_title", ""),
                "final_recipe": {
                    "ingredients": recipe_result.get("ingredients", []) or [],
                    "steps": recipe_result.get("steps", []) or [],
                    "tips": ["LLM 출력 파싱 실패 - 원본 레시피를 그대로 사용하세요."]
                },
                "missing_ingredients": [],
                "substitutes": {}
            }),
            "_raw_llm_output": raw,
        }

    parsed["final_text"] = _render_final_text(parsed)
    return parsed
