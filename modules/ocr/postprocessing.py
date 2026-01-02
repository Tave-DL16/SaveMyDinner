from __future__ import annotations

import ast
from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL_CACHE: dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    cached = _MODEL_CACHE.get(model_name)
    if cached:
        return cached
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _build_prompt(ocr_list: Sequence[str]) -> str:
    return f"""
다음은 OCR에서 추출된 텍스트 리스트다. 매우 많은 노이즈가 있다.
너의 작업:
1) 식자재 이름만 추출
2) 원재료명, 가공식품명, 식품 카테고리는 포함
3) 랜덤문자, 제조코드, 날짜, 숫자, 브랜드명, meaningless token 제거
4) 사소한 오타는 보정하여 올바른 식자재명으로 정규화할 것
5) 최종 결과는 파이썬 리스트만 출력
6) 사람들이 흔하게 먹지 않는 식자재는 제외 예시) 치자, 호박잎
7) 중복되는 식자재는 제외 만약 체다 치즈, 체다, 이런식으로 여러가지가 있다면 가장 세부적인 체다 치즈만 포함

아래는 예시다. (COT Reasoning 포함)

### 예시 1
입력:
["Cheese","P","00393","CHEDDAR","20배입","CEEDCHEESE","S00TFi0"]

모델 사고 과정(COT 요약):
- "Cheese", "CHEDDAR"는 의미 있음
- "20배입"은 치즈 구성 정보이므로 식자재명 아님
- "CEEDCHEESE"는 'Cheese' 오타 -> 보정
- 나머지 랜덤 코드/문자 제거

정답:
["Cheddar Cheese"]

### 예시 2
입력:
["김밥햄","국산무사용","김밥단무지","두번구워","WZ0105302","1150","MB6ZE"]

모델 사고 과정(COT 요약):
- "김밥햄" -> yes
- "김밥단무지" -> yes
- "국산무사용" -> 원재료 특성 정보 -> 제외
- "두번구워" -> 조리 방식 -> 제외
- 나머지는 코드/번호 -> 제거

정답:
["김밥햄", "김밥단무지"]

--------------------------
### 이제 아래 데이터를 처리하라:

입력:
{list(ocr_list)}

마지막에 다른 부가적인 내용없이 오직 **파이썬 리스트 형태의 결과만 출력**하라.
"""


def _extract_list(text: str) -> List[str]:
    left = text.find("[")
    right = text.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return []
    snippet = text[left : right + 1]
    try:
        value = ast.literal_eval(snippet)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clean_ocr_with_llm(
    ocr_list: Sequence[str],
    model_name: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 2048,
) -> List[str]:
    print(f"   LLM 입력 ({len(ocr_list)}개 텍스트): {ocr_list[:20]}...")  # 처음 20개만

    tokenizer, model = _load_model(model_name)
    prompt = _build_prompt(ocr_list)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(f"   LLM 원본 출력: {content}")

    result = _extract_list(content)
    print(f"   LLM 파싱 결과: {result}")

    return result
