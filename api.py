from fastapi import FastAPI
from pydantic import BaseModel
from inference import classify_email   # your function
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Spam Filter API (Qwen-0.5B LoRA)",
    description="CPU-optimized spam classifier",
    version="1.0.0"
)

# Allow UI (Streamlit, Angular, etc.) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailInput(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    label: str      # "spam" or "ham"
    confidence: float
    explanation: str


@app.post("/classify", response_model=ClassificationResult)
def classify(input_data: EmailInput):
    """
    Classify email/text content into spam/ham.
    """
    result = classify_email(input_data.text)

    return ClassificationResult(
        label=result["label"],
        confidence=result["confidence"],
        explanation=result["explanation"]
    )
