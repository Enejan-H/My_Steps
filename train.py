from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

# -----------------------------
# 1. Dataset
# -----------------------------
dataset = load_dataset("yelp_review_full", split="train[:1%]")  # küçük bir parça

# -----------------------------
# 2. Model & Tokenizer
# -----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# -----------------------------
# 3. Tokenize function
# -----------------------------
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# 4. Training arguments
# -----------------------------
output_dir = "./finetuned_model_final"
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=10,
)

# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# -----------------------------
# 6. Start training
# -----------------------------
trainer.train()

# -----------------------------
# 7. Save model & tokenizer (inference-ready)
# -----------------------------
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Model ve tokenizer '{output_dir}' klasörüne kaydedildi. Artık inference-ready!")
