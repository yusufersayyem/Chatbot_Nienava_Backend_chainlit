import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Search Backend")

HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def _process_response(self, response) -> List[float]:
        if hasattr(response, "tolist"):
            response = response.tolist()
        while isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            if isinstance(response[0][0], list):
                response = response[0]
            else:
                break
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            response = [sum(col) / len(response) for col in zip(*response)]
        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text)
        return self._process_response(response)

embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

class QueryRequest(BaseModel):
    query: str

@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    docs = vector_store.similarity_search(req.query, k=1)
    
    async def event_generator():
        if docs:
            full_text = docs[0].page_content
            # تقسيم النص إلى كلمات لتتدفق تدريجياً
            words = full_text.split(" ")
            for word in words:
                yield {"data": word + " "}
                await asyncio.sleep(0.04)  # سرعة إخراج الكلمات (يمكنك تعديلها)
        else:
            yield {"data": "عذراً، هذه المعلومة غير متوفرة في قاعدة البيانات المتاحة لدي."}

    return EventSourceResponse(event_generator())
