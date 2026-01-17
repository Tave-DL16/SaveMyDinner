from ultralytics.models.sam import SAM3SemanticPredictor
from transformers import AutoModelForCausalLM, Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights" / "sam3.pt"

def load_qwen3_vl_quantized():
    """
    Load 4-bit quantized Qwen3-VL-8B model.
    This model will be shared for both VLM and LLM tasks to save memory.
    """
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    
    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model, processor

def load_varco_ocr():
    from transformers import LlavaOnevisionForConditionalGeneration
    
    model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    
    # 4-bit quantization for VARCO OCR as well
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

def load_varco_ocr():
    from transformers import LlavaOnevisionForConditionalGeneration
    
    model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    
    # 4-bit quantization for VARCO OCR
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor

def load_sam3():
    overrides = dict(
    conf=0.75,
    task="segment",
    mode="predict",
    model=str(WEIGHTS_DIR),
    half=True,  # Use FP16 for faster inference
    save=False,
    )
    SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)
    return SAM_MODEL