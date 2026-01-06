from pathlib import Path

from paddleocr import PaddleOCR
from ultralytics.models.sam import SAM3SemanticPredictor

from modules.ocr.ocr_inference import run_ocr_with_rotations
from modules.ocr.detect_ingredients import run_sam
from modules.ocr.qwen_model import postprocessing_with_vlm,clean_ocr_with_llm

def _dedupe_keep_order(items):
    """리스트 아이템 중복제거 함수"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "pngtree-fresh-vegetables-set-realistic-tomatoes-cucumber-carrots-lettuce-in-ultra-hd-png-image_17699620.webp"
OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True)


overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str(BASE_DIR / "sam3.pt"),
    half=True,  # Use FP16 for faster inference
    save=True,
)
SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)

def run_ocr_pipeline(
        ocr_engine,
        sam_model,
        image_path,
        model_name_llm = "Qwen/Qwen3-1.7B",
        model_name_vlm = "Qwen/Qwen3-VL-2B-Instruct",
        rotations = [0,90,180,270]
):
    print("step1 OCR")
    raw_texts = run_ocr_with_rotations(ocr_engine, image_path, rotations=rotations)
    print(raw_texts)

    print("Step2 TEXT LLM 후처리")
    llm_items = clean_ocr_with_llm(raw_texts, model_name = model_name_llm)
    print(llm_items)

    print("Step3 SAM_MODEL")
    sam_result = run_sam(image_path=image_path, model=sam_model)
    print(sam_result)

    print("Step4 VLM 후처리")
    vlm_items = postprocessing_with_vlm(sam_item=sam_result, model_name=model_name_vlm)
    
    final_result = [*llm_items , *vlm_items]

    return _dedupe_keep_order(final_result)


if __name__ == "__main__":
    result = run_ocr_pipeline(
        ocr_engine=OCR_ENGINE,
        sam_model=SAM_MODEL,
        image_path=IMAGE_PATH,
    )
    print(f"\nFinal result: {result}")
