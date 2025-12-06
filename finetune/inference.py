from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "./finetuned_model_final"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

text = "I love this product!"

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred = torch.argmax(logits, dim=-1).item()
print("Prediction:", pred)
