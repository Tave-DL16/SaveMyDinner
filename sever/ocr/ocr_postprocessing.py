import json
import re
import ast
from pathlib import Path

def ocr_postprocessing(
        item_list,
        model,
        processor
):
    # Prompt loading
    prompt_path = Path(__file__).parent.parent / "prompts" / "ocr_postprocessing.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    print(f"Starting OCR Post-processing for {len(item_list)} items with Qwen3-VL (batch mode)...")

    # Skip if empty list
    if not item_list or len(item_list) == 0:
        return []
    
    # Format the OCR list as a string
    ocr_list_str = "\n".join([f"- {item}" for item in item_list if item and item.strip()])
    
    # Format prompt with entire list
    prompt = base_prompt.replace("{ocr_list}", ocr_list_str)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Use processor for VL model - need to use tokenizer for text-only
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize using processor's tokenizer
    model_inputs = processor.tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,  # Increased for longer list
    )
    
    # Extract new tokens and decode
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    generated_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    print(f"Model output: {generated_text}")
    
    # Parse Python list output
    try:
        # Try to find list in output
        list_match = re.search(r'\[.*?\]', generated_text, re.DOTALL)
        if list_match:
            list_str = list_match.group(0)
            # Safely evaluate the list
            import ast
            cleaned_ingredients = ast.literal_eval(list_str)
            
            # Ensure it's a list of strings
            if isinstance(cleaned_ingredients, list):
                cleaned_ingredients = [str(item) for item in cleaned_ingredients if item]
                print(f"Extracted {len(cleaned_ingredients)} valid ingredients: {cleaned_ingredients}")
                return cleaned_ingredients
            else:
                print(f"Output is not a list: {cleaned_ingredients}")
                return []
        else:
            print(f"Failed to find list in output: {generated_text}")
            return []
            
    except Exception as e:
        print(f"Error parsing output: {e}. Output: {generated_text}")
        return []

