import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI(
    title="Optimized Sentiment Analysis API",
    description="Distilled & Quantized ONNX Model Deployment"
)

# Load Tokenizer, Config, and ONNX Session
quantized_folder = os.path.abspath("onnx_quantized_model")
quantized_onnx_file = os.path.join(quantized_folder, "model_quantized.onnx")

tokenizer = AutoTokenizer.from_pretrained(quantized_folder)
session = ort.InferenceSession(quantized_onnx_file)
expected_inputs = {input_node.name for input_node in session.get_inputs()}

# Explicit sentiment label mapping
LABEL_MAP = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    logits: list

@app.get("/")
def health_check():
    return {"status": "online", "model": "Distilled INT8 ONNX"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    # Tokenize input
    inputs = tokenizer(request.text, return_tensors="np")
    onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in expected_inputs}
    
    # Run ONNX inference
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0][0]
    
    # Calculate probabilities and mapped label
    probs = softmax(logits)
    predicted_class = int(np.argmax(logits))
    predicted_label = LABEL_MAP.get(predicted_class, f"CLASS_{predicted_class}")
    confidence = float(probs[predicted_class])
    
    return PredictResponse(
        text=request.text,
        label=predicted_label,
        confidence=round(confidence, 4),
        logits=logits.tolist()
    )