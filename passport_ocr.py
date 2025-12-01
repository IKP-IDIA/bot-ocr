from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import re
import fitz  # PyMuPDF
import os
import json
from typing import Dict, List, Optional
import torch
from contextlib import asynccontextmanager
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
processor = None
model_loading = False
model_load_error = None

async def load_typhoon_model_async():
    """Load Typhoon OCR model asynchronously"""
    global model, processor, model_loading, model_load_error
    
    if model is not None:
        return
    
    if model_loading:
        return
    
    model_loading = True
    
    try:
        logger.info("Starting Typhoon OCR model loading...")
        model_name = "scb10x/typhoon-ocr1.5-2b"
        
        # Determine device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        logger.info(f"Using device: {device}, dtype: {dtype}")
        
        # Load processor first (faster)
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with optimizations
        logger.info("Loading model (this may take 2-5 minutes)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Optimize memory usage
        ).eval()
        
        # Optimize for inference
        if torch.cuda.is_available():
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Model loaded on CPU - inference will be slower")
            # For CPU, we can try to optimize further
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        logger.info("Model loaded successfully!")
        model_loading = False
        
    except Exception as e:
        model_load_error = str(e)
        model_loading = False
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

def resize_if_needed(img, max_size=1800):
    """Resize image if needed (Typhoon OCR is trained with 1800px max)"""
    width, height = img.size
    if width > max_size or height > max_size:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, int(height * scale))
        else:
            scale = max_size / float(height)
            new_size = (int(width * scale), max_size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized from {(width, height)} to {img.size}")
    return img

def get_typhoon_prompt():
    """Get the standard Typhoon OCR prompt for v1.5"""
    return """Below is an image of a document page.

The available sub-tasks for extracting content are:
- For 'text' parts: Extract readable text, keeping the original language (Thai or English) and structure (e.g., paragraphs, bullet points)
- For 'figure' parts: Provide meaningful alt-text. Describe any text in the figure (labels, captions) and the visual elements shown (e.g., "A bar chart comparing sales across quarters"). 
- For 'table' parts: Convert to markdown. Be precise. If OCR quality is low, note it. 

Extract the content of all document elements in a structured format.

**output as JSON format**"""

def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using Typhoon OCR"""
    try:
        # Resize image if needed
        image = resize_if_needed(image)
        
        # Prepare messages
        prompt = get_typhoon_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # Apply chat template and process vision info
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate with timeout protection
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=16000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Parse JSON output if possible
        try:
            # Remove markdown code blocks if present
            clean_text = output_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            elif clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "").strip()
            
            result = json.loads(clean_text)
            # Extract natural_text from the result if available
            if isinstance(result, dict) and 'natural_text' in result:
                return result['natural_text']
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text
            return output_text
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"OCR extraction failed: {str(e)}")

def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """Convert PDF pages to images"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Render page to image at 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
    
    pdf_document.close()
    return images

def identify_passport_fields(text: str) -> Dict[str, any]:
    """Identify name, surname, and passport number from extracted text"""
    
    result = {
        "has_name": False,
        "has_surname": False,
        "has_passport_number": False,
        "name": None,
        "surname": None,
        "passport_number": None,
        "raw_text": text
    }
    
    # Convert to uppercase for pattern matching
    text_upper = text.upper()
    lines = text.split('\n')
    
    # Pattern for passport number (common formats)
    passport_patterns = [
        r'\b[A-Z]{1,2}\d{6,9}\b',  # Letter(s) followed by digits
        r'\b\d{8,9}\b',  # Pure digits
        r'(?:PASSPORT\s*(?:NO|NUMBER|NUM)?[:\s]*)?([A-Z]{1,2}\d{6,9})',
        r'(?:PASSPORT\s*(?:NO|NUMBER|NUM)?[:\s]*)?(\d{8,9})'
    ]
    
    # Search for passport number
    for pattern in passport_patterns:
        match = re.search(pattern, text_upper)
        if match:
            passport_num = match.group(1) if match.lastindex else match.group(0)
            result["passport_number"] = passport_num.strip()
            result["has_passport_number"] = True
            break
    
    # Common passport field keywords
    name_keywords = ['GIVEN NAME', 'GIVEN NAMES', 'FIRST NAME', 'NAME', 'PRENOM', 'GIVEN']
    surname_keywords = ['SURNAME', 'LAST NAME', 'FAMILY NAME', 'NOM', 'SUR NAME']
    
    # Search for name and surname
    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        
        # Check for surname
        for keyword in surname_keywords:
            if keyword in line_upper:
                parts = re.split(r'[:/]', line)
                if len(parts) > 1:
                    surname = parts[-1].strip()
                    if surname and len(surname) > 1 and not surname.upper() in surname_keywords:
                        result["surname"] = surname
                        result["has_surname"] = True
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and len(next_line) > 1:
                        result["surname"] = next_line
                        result["has_surname"] = True
                break
        
        # Check for given name
        for keyword in name_keywords:
            if keyword in line_upper and 'SURNAME' not in line_upper:
                parts = re.split(r'[:/]', line)
                if len(parts) > 1:
                    name = parts[-1].strip()
                    if name and len(name) > 1 and not name.upper() in name_keywords:
                        result["name"] = name
                        result["has_name"] = True
                elif i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and len(next_line) > 1:
                        result["name"] = next_line
                        result["has_name"] = True
                break
    
    # Look for MRZ (Machine Readable Zone)
    mrz_lines = [line for line in lines if len(line) > 30 and re.match(r'^[A-Z0-9<]+$', line.strip())]
    if mrz_lines and not (result["has_name"] and result["has_surname"]):
        for mrz in mrz_lines[1:2]:
            parts = mrz.split('<<')
            if len(parts) >= 2:
                if not result["has_surname"]:
                    result["surname"] = parts[0].replace('<', ' ').strip()
                    result["has_surname"] = True
                if not result["has_name"]:
                    result["name"] = parts[1].split('<')[0].replace('<', ' ').strip()
                    result["has_name"] = True
    
    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    # Start loading model in background - don't wait for it
    asyncio.create_task(load_typhoon_model_async())
    logger.info("Model loading initiated in background")
    yield
    # Cleanup if needed
    logger.info("Shutting down...")

app = FastAPI(title="Passport OCR API", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "Passport OCR API with Typhoon",
        "status": "running",
        "model": "typhoon-ocr1.5-2b",
        "model_status": "loaded" if model is not None else ("loading" if model_loading else "not_loaded"),
        "endpoints": {
            "POST /extract": "Extract text and identify passport fields from image/PDF",
            "GET /health": "Check API health and model status",
            "POST /load-model": "Manually trigger model loading"
        }
    }

@app.post("/load-model")
async def trigger_model_load(background_tasks: BackgroundTasks):
    """Manually trigger model loading"""
    if model is not None:
        return {"status": "already_loaded"}
    
    if model_loading:
        return {"status": "loading_in_progress"}
    
    background_tasks.add_task(load_typhoon_model_async)
    return {"status": "loading_started"}

@app.post("/extract")
async def extract_passport(file: UploadFile = File(...)):
    """
    Extract text from passport and identify key fields
    
    Supports: PDF, PNG, JPG, JPEG files
    """
    
    # Check model status
    if model is None:
        if model_loading:
            raise HTTPException(
                status_code=503, 
                detail="Model is still loading. Please try again in a few moments."
            )
        elif model_load_error:
            raise HTTPException(
                status_code=503,
                detail=f"Model failed to load: {model_load_error}"
            )
        else:
            # Try to start loading
            asyncio.create_task(load_typhoon_model_async())
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Loading has been initiated. Please try again in 2-3 minutes."
            )
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file
        file_bytes = await file.read()
        
        # Process based on file type
        images = []
        if file_ext == '.pdf':
            images = pdf_to_images(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images = [image]
        
        # Extract text from all images
        all_text = []
        all_results = []
        
        for idx, image in enumerate(images):
            logger.info(f"Processing page {idx + 1}/{len(images)}...")
            extracted_text = extract_text_from_image(image)
            all_text.append(extracted_text)
            
            # Identify passport fields
            fields = identify_passport_fields(extracted_text)
            fields['page'] = idx + 1
            all_results.append(fields)
        
        # Combine results from all pages
        combined_result = {
            "total_pages": len(images),
            "has_name": any(r["has_name"] for r in all_results),
            "has_surname": any(r["has_surname"] for r in all_results),
            "has_passport_number": any(r["has_passport_number"] for r in all_results),
            "name": next((r["name"] for r in all_results if r["name"]), None),
            "surname": next((r["surname"] for r in all_results if r["surname"]), None),
            "passport_number": next((r["passport_number"] for r in all_results if r["passport_number"]), None),
            "all_pages": all_results,
            "full_text": "\n\n--- Page Break ---\n\n".join(all_text)
        }
        
        return JSONResponse(content=combined_result)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "model_load_error": model_load_error,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "10.1.0.150")
    uvicorn.run(app, host=host, port=port)
