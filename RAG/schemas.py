from pydantic import BaseModel
from typing import List

class Query(BaseModel):
    query: str = ""

class RagResponse(BaseModel):
    context: List[str] = []
    response: str = ""

