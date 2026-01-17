from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import sys
import os

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from server.load_model import load_qwen3_vl_quantized, load_sam3, load_varco_ocr
from ocr.total_pipeline import run_total_pipeline

app = FastAPI(title="Ingredient Detection API")

# Global model storage
models = {}

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    print("Loading models...")
    
    # Load SAM model
    print("Loading SAM3...")
    models["sam"] = load_sam3()
    
    # Load quantized Qwen3-VL-8B (shared for VLM and LLM)
    print("Loading Qwen3-VL-8B (4-bit quantized)...")
    shared_model, shared_processor = load_qwen3_vl_quantized()
    
    # Use same model for VLM and LLM tasks
    models["vlm_model"] = shared_model
    models["vlm_processor"] = shared_processor
    models["llm_model"] = shared_model
    models["llm_processor"] = shared_processor
    
    # Load VARCO OCR (separate specialized model)
    print("Loading VARCO OCR (4-bit quantized)...")
    models["ocr_model"], models["ocr_processor"] = load_varco_ocr()
    
    print("All models loaded successfully!")

@app.post("/detect-ingredients")
async def detect_ingredients(file: UploadFile = File(...)):
    """
    POST endpoint to detect ingredients from an uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
    
    Returns:
        JSON with list of detected ingredients
    """
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / file.filename
        
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run the pipeline
        ingredients = run_total_pipeline(
            image_path=temp_file_path,
            sam_model=models["sam"],
            vlm_model=models["vlm_model"],
            vlm_processor=models["vlm_processor"],
            ocr_model=models["ocr_model"],
            ocr_processor=models["ocr_processor"],
            llm_model=models["llm_model"],
            llm_processor=models["llm_processor"]
        )
        
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        return JSONResponse(content={
            "success": True,
            "ingredients": ingredients,
            "count": len(ingredients)
        })
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(models) > 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
