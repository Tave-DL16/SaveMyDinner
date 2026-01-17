from pathlib import Path
import torch

from ocr.detect_ingredients import run_sam
from ocr.vlm_postprocessing import vlm_postprocessing
from ocr.ocr import run_varco_ocr
from ocr.ocr_postprocessing import ocr_postprocessing
from ocr.translate import translate_ingredients


def run_total_pipeline(
    image_path: Path,
    sam_model,
    vlm_model,
    vlm_processor,
    ocr_model,
    ocr_processor,
    llm_model,
    llm_processor
):
    """
    Runs the full ingredient detection pipeline:
    1. SAM -> Crop ingredients
    2. VLM -> Identify cropped ingredients
    3. OCR -> Extract text from image
    4. LLM -> Filter and correct OCR text
    5. Combine results
    """
    
    print(f"Processing image: {image_path}")
    
    # 1. SAM & VLM (Visual detection)
    print("Running SAM (Visual Detection)...")
    crops = run_sam(image_path, sam_model)
    
    vlm_ingredients = []
    print(f"Running VLM on {len(crops)} crops...")
    for i, crop in enumerate(crops):
        try:
            # vlm_postprocessing returns a list of strings, usually one item
            result = vlm_postprocessing(crop, vlm_model, vlm_processor)
            if result and len(result) > 0:
                name = result[0].strip()
                print(f"  Crop {i}: {name}")
                vlm_ingredients.append(name)
        except Exception as e:
            print(f"  Error processing crop {i}: {e}")

    # Translate VLM results if any (Translate English to Korean one-by-one)
    if vlm_ingredients:
        vlm_ingredients = translate_ingredients(vlm_ingredients)
        print(f"  Translated VLM results: {vlm_ingredients}")
      

    # 2. OCR & LLM (Text detection)
    print("Running OCR (Text Detection with VARCO OCR)...")
    ocr_texts = run_varco_ocr(image_path, ocr_model, ocr_processor)
    print(f"  OCR extracted {(ocr_texts)} text fragments.")
    
    print("Running OCR Post-processing (LLM)...")
    ocr_ingredients = ocr_postprocessing(ocr_texts, llm_model, llm_processor)
    print(f"  OCR identified {len(ocr_ingredients)} ingredients: {ocr_ingredients}")

    # 3. Combine Results
    # Use set to remove duplicates, sort for consistency
    all_ingredients = sorted(list(set(vlm_ingredients + ocr_ingredients)))
    
    print(f"Final Ingredient List: {all_ingredients}")
    
    return all_ingredients


