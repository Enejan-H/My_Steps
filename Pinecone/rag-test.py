from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
load_dotenv()

YOUR_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=YOUR_API_KEY, environment="us-east1-gcp")

index_name = "rag-test-py"

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

# 4) Embedding modeli yükle
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en")

# 5) Upsert
docs = [
    ("id1", "FastAPI is a Python web framework."),
    ("id2", "Docker containers are used for packaging apps."),
]
vectors = [(doc_id, model.encode(text).tolist(), {"text": text}) for doc_id, text in docs]
index.upsert(vectors)

# 6) Query
query = "What is Docker container?"
q_vec = model.encode(query).tolist()
res = index.query(vector=q_vec, top_k=2, include_metadata=True)
print(res)