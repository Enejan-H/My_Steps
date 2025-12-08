from fastapi import FastAPI
from dotenv import load_dotenv
import os
from rag import get_embedding, generate_answer_hf, retrieve_context_with_vector
from schemas import Query, RagResponse
from utils.logger import setup_logger
from pinecone import Pinecone

load_dotenv()
logger = setup_logger()
app = FastAPI()

pinecone_client = None
pinecone_index = None

@app.on_event("startup")
def startup_event():
    global pinecone_client, pinecone_index
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found")

    try:
        pinecone_client = Pinecone(api_key=api_key)
        pinecone_index = pinecone_client.Index("rag-demo")
        logger.info("Pinecone initialized successfully")
    except Exception as e:
        logger.error(f"Pinecone init error: {e}")
        pinecone_client = None
        pinecone_index = None


@app.post("/rag", response_model=RagResponse)
async def rag_endpoint(body: Query):
    query = body.query or ""
    vector = get_embedding(query)  # HuggingFace embedding
    context = await retrieve_context_with_vector(vector, pinecone_index)
    response = generate_answer_hf(query, context)  # HuggingFace LLM
    return RagResponse(context=context, response=response)
