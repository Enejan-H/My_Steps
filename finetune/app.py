import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------------
# 1. Load model + tokenizer
# -----------------------------
checkpoint = "./finetuned_model_final"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Sentiment Classifier")

text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        st.success(f"Prediction: {pred}")
