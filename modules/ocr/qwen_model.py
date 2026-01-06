from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

Purpose = Literal["ocr", "sam"]


@dataclass
class ModelBundle:
    model: object
    tokenizer: Optional[object] = None
    processor: Optional[object] = None


class QwenPromptManager:
    def __init__(
        self,
        base_dir: Path = Path("modules/ocr/prompts"),
        mapping: Optional[Dict[Purpose, str]] = None,
    ) -> None:
        self.base_dir = base_dir
        self.mapping = mapping or {
            "ocr": "ocr_prompt.txt",
            "sam": "sam_propmt.txt",
        }
        self._cache: Dict[Purpose, str] = {}

    def get(self, purpose: Purpose, **kwargs: object) -> str:
        template = self._load_template(purpose)
        if "ocr_list" in kwargs:
            template = template.replace(
                "{list(ocr_list)}",
                str(list(kwargs["ocr_list"])),
            )

        for key, value in kwargs.items():
            token = "{" + key + "}"
            if token in template:
                template = template.replace(token, str(value))

        return template

    def _load_template(self, purpose: Purpose) -> str:
        cached = self._cache.get(purpose)
        if cached:
            return cached

        filename = self.mapping.get(purpose)
        if not filename:
            raise ValueError(f"Unsupported purpose: {purpose}")
        path = self.base_dir / filename
        text = path.read_text(encoding="utf-8")
        self._cache[purpose] = text
        return text


class QwenModelLoader:
    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-1.7B",
        use_vision: bool = False,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        prompt_manager: Optional[QwenPromptManager] = None,
    ) -> None:
        self.model_name = model_name
        self.use_vision = use_vision
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.prompts = prompt_manager or QwenPromptManager()
        self._bundle: Optional[ModelBundle] = None

    def load(self) -> ModelBundle:
        if self._bundle:
            return self._bundle

        if self.use_vision:
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self._bundle = ModelBundle(model=model, processor=processor)
            return self._bundle

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self._bundle = ModelBundle(model=model, tokenizer=tokenizer)
        return self._bundle

    def get_prompt(self, purpose: Purpose, **kwargs: object) -> str:
        return self.prompts.get(purpose, **kwargs)


_MODEL_CACHE: Dict[Tuple[str, bool], ModelBundle] = {}
_PROMPTS = QwenPromptManager()


def _load_model(model_name: str, use_vision: bool = False) -> ModelBundle:
    cache_key = (model_name, use_vision)
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    if use_vision:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        bundle = ModelBundle(model=model, processor=processor)
        _MODEL_CACHE[cache_key] = bundle
        return bundle

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    bundle = ModelBundle(model=model, tokenizer=tokenizer)
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _generate_text(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int,
) -> str:
    tokenizer = bundle.tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer is required for text generation.")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(bundle.model.device)
    generated_ids = bundle.model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def _load_image(value: Union[Image.Image, Path, str]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.open(value).convert("RGB")


def _generate_vlm_text(
    bundle: ModelBundle,
    prompt: str,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    processor = bundle.processor
    if processor is None:
        raise ValueError("Processor is required for vision generation.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        text = prompt

    model_inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        bundle.model.device
    )
    generated_ids = bundle.model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    if hasattr(processor, "tokenizer"):
        return processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip("\n")


def _extract_list(text: str) -> Optional[List[str]]:
    left = text.find("[")
    right = text.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return None
    snippet = text[left : right + 1]
    try:
        value = ast.literal_eval(snippet)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, list):
        return None
    items: List[str] = []
    for item in value:
        text = str(item).strip()
        if not text or text in {"[]", "[ ]"}:
            continue
        items.append(text)
    return items


def clean_ocr_with_llm(
    ocr_list: Iterable[str],
    model_name: str = "Qwen/Qwen3-1.7B",
    max_new_tokens: int = 2048,
) -> Optional[List[str]]:
    bundle = _load_model(model_name, use_vision=False)
    prompt = _PROMPTS.get("ocr", ocr_list=list(ocr_list))
    content = _generate_text(bundle, prompt, max_new_tokens=max_new_tokens)
    parsed = _extract_list(content)
    if parsed is not None:
        return parsed
    cleaned = content.strip()
    if cleaned in {"[]", "[ ]"}:
        return []
    return None


def postprocessing_with_vlm(
    sam_item: Sequence[Union[Image.Image, Path, str]],
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    max_new_tokens: int = 512,
) -> List[str]:
    bundle = _load_model(model_name, use_vision=True)
    prompt = _PROMPTS.get("sam")

    results: List[str] = []
    for item in sam_item:
        image = _load_image(item)
        content = _generate_vlm_text(
            bundle,
            prompt,
            image=image,
            max_new_tokens=max_new_tokens,
        )
        answer = content.strip()
        results.append(answer if answer else "")
    return results
