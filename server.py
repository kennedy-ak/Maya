"""
FastAPI Server for Ghana Sexual Health Chatbot
Provides REST API endpoints for the chatbot model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import asyncio
from typing import Optional
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

LORA_ADAPTER_PATH = os.getenv("MODEL_PATH", "ghana_contraception_lora")
BASE_MODEL = "unsloth/Llama-3.2-3B"
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_SEQ_LENGTH = 2048

# ============================================================================
# INITIALIZE FASTAPI
# ============================================================================

app = FastAPI(
    title="Ghana Sexual Health Chatbot API",
    description="API for answering questions about contraception, STIs, and reproductive health in Ghana",
    version="1.0.0"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL MODEL VARIABLES
# ============================================================================

model = None
tokenizer = None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class QuestionResponse(BaseModel):
    question: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the model on startup"""
    global model, tokenizer

    print("🔄 Loading model...")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   LoRA adapter: {LORA_ADAPTER_PATH}")

    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH, token=HF_TOKEN)

        # Load base model
        print("📦 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=HF_TOKEN,
        )

        # Apply LoRA adapter
        print("🔧 Applying LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

        # Merge weights
        print("⚡ Merging LoRA weights...")
        model = model.merge_and_unload()
        model.eval()

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

async def generate_answer(question: str, temperature: float = 0.7, max_tokens: int = 256):
    """Generate answer for a question"""

    # Format prompt
    prompt = f"""Below is a question about contraception in Ghana. Write a helpful, accurate response.

### Question:
{question}

### Response:
"""

    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )

    # Decode
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract answer
    if "### Response:" in full_response:
        answer = full_response.split("### Response:")[-1].strip()
    else:
        answer = full_response.split(question)[-1].strip()

    return answer

async def generate_answer_stream(question: str, temperature: float = 0.7, max_tokens: int = 256):
    """Generate answer with streaming (token by token)"""

    prompt = f"""Below is a question about contraception in Ghana. Write a helpful, accurate response.

### Question:
{question}

### Response:
"""

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate with streaming
    for i in range(max_tokens):
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=True,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        # Get the new token
        new_token = outputs[0][-1:]
        token_text = tokenizer.decode(new_token, skip_special_tokens=True)

        # Yield token
        yield f"data: {json.dumps({'token': token_text})}\n\n"

        # Update inputs for next iteration
        inputs = {"input_ids": outputs}

        # Stop if we hit end token
        if tokenizer.eos_token_id in new_token:
            break

        await asyncio.sleep(0)  # Allow other tasks to run

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface"""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Ghana Sexual Health Chatbot API</h1>
                <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>To use the chat interface, create a static/index.html file.</p>
            </body>
        </html>
        """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer"""

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = await generate_answer(
            request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """Ask a question and stream the answer token by token"""

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return StreamingResponse(
        generate_answer_stream(
            request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        ),
        media_type="text/event-stream"
    )

# ============================================================================
# MOUNT STATIC FILES
# ============================================================================

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
