from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os   
load_dotenv()
# HuggingFace API anahtarı (gerekirse)
hf_key = os.getenv("HF_API_KEY")

# -----------------------------
# Embedding modeli
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embedding_model.encode(text).tolist()


# -----------------------------
# LLM modeli
# -----------------------------
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",    # GPU varsa GPU
    dtype=torch.float32
)

def generate_answer_hf(query: str, context: list[str], max_new_tokens=50):
    context_str = "\n".join(context)
    #prompt = f"Context:\n{context_str}\n\nQuery: {query}\nAnswer:"
    prompt = f"Answer the question ONLY using the following context. Do not use any external knowledge.\n{context_str}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    output = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# Pinecone retrieval
# -----------------------------
async def retrieve_context_with_vector(vector, index, top_k=3):
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    contexts = [m["metadata"]["text"] for m in result["matches"]]
    return contexts
