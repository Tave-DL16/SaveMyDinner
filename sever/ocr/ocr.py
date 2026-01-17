import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re


def run_varco_ocr(
    image_path,
    model,
    processor,
    angle_list=[0]
):
    """
    Run VARCO OCR on an image with multiple rotation angles.
    
    Args:
        image_path: Path to the image file
        model: VARCO OCR model
        processor: VARCO OCR processor
        angle_list: List of rotation angles to try
    
    Returns:
        List of detected text strings
    """
    char_list = []
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    
    # Image upscaling for OCR performance boost
    w, h = image_pil.size
    target_size = 2304
    if max(w, h) < target_size:
        scaling_factor = target_size / max(w, h)
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        image_pil = image_pil.resize((new_w, new_h))
    
    # Convert to cv2 for rotation
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Run OCR on different angles
    for angle in angle_list:
        if angle == 0:
            target_pil = image_pil
        else:
            target = rotate_image(image, angle)
            target_pil = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": target_pil},
                    {"type": "text", "text": "<ocr>"},
                ],
            },
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)
        
        generate_ids = model.generate(**inputs, max_new_tokens=1024)
        
        generate_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        
        output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)
        
        # Extract text from <char> tags
        char_list.extend(re.findall(r"<char>(.*?)</char>", output))
    
    return char_list


def rotate_image(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate while keeping the full canvas."""
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    rad = np.deg2rad(angle)
    new_w = int(abs(h * np.sin(rad)) + abs(w * np.cos(rad)))
    new_h = int(abs(h * np.cos(rad)) + abs(w * np.sin(rad)))
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    m[0, 2] += new_w / 2 - center[0]
    m[1, 2] += new_h / 2 - center[1]
    return cv2.warpAffine(
        image_bgr,
        m,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

