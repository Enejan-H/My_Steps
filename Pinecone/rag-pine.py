# rag_pipeline_with_llm.py

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
load_dotenv()
import torch

# ----------------------------
# 1️⃣ .env ve API key
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY yok!")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY yok!")

# ----------------------------
# 2️⃣ Pinecone setup
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-demo"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

index = pc.Index(index_name)
# ----------------------------
# 3️⃣ Embedding model
# ----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en", device=device)

# ----------------------------
# 4️⃣ Örnek dokümanları upsert et
# ----------------------------
docs = [
    ("id1", "FastAPI is a Python web framework."),
    ("id2", "Docker containers are used for packaging apps."),
    ("id3", "RAG stands for Retrieval-Augmented Generation, combining retrieval with LLMs."),
    ("id4", "Pinecone is a vector database for machine learning applications."),
    ("id5", "Sentence Transformers provide an easy way to compute dense vector representations for sentences.")
]

vectors = [(doc_id, model.encode(text).tolist(), {"text": text}) for doc_id, text in docs]
index.upsert(vectors)

# ----------------------------
# 5️⃣ Kullanıcı query
# ----------------------------
query = "What is AI?"
query_embedding = model.encode(query).tolist()

# ----------------------------
# 6️⃣ Pinecone query → top_k
# ----------------------------
top_k = 2
res = index.query(
    index_name=index_name,
    vector=query_embedding,
    top_k=top_k,
    include_metadata=True
)

# ----------------------------
# 7️⃣ Context oluştur
# ----------------------------
context = "\n".join([match['metadata']['text'] for match in res['matches']])

# ----------------------------
# 8️⃣ Prompt oluştur
# ----------------------------
prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

# ----------------------------
# 9️⃣ LLM çağrısı
# ----------------------------
# ----------------------------
# 9️⃣ LLM çağrısı (Flan-T5)
# ----------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

prompt = "Answer based on context:\n" + context + "\nQuestion: " + query
answer = llm_pipe(prompt)[0]['generated_text']

print("\nContext from Pinecone:\n", context)
print("\nLLM Answer:\n", answer)
