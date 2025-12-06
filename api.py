from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------
# 1. Load model & tokenizer
# -----------------------------
checkpoint = "./finetuned_model_final"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

app = FastAPI(title="Sentiment Classifier API")

# -----------------------------
# 2. Request schema
# -----------------------------
class TextRequest(BaseModel):
    text: str

# -----------------------------
# 3. Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()
    return {"prediction": pred}
