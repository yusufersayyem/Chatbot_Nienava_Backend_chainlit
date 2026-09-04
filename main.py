import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sse_starlette.sse import EventSourceResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model: SentenceTransformer = None
loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None

def load_data():
    global model, loaded_chunks, chunk_embeddings
    
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("❌ ملفات المتجهات غير موجودة!")

    print("⚡ جاري تحميل المتجهات من النواة المحلية...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print("⏳ جاري تحميل نموذج ONNX الخفيف تقليلاً للذاكرة...")
    # استخدام backend="onnx" يمنع استهلاك الذاكرة العالي بفضل محرك ONNX
    model = SentenceTransformer(MODEL_NAME, backend="onnx")
    print("✅ تم تحميل الباك إند بنجاح وبأقل استهلاك ذاكرة ممكن!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield

app = FastAPI(title="Nineveh Edu Search - Low Memory", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def semantic_search(query: str, top_k: int = 3) -> List[str]:
    if model is None or chunk_embeddings is None:
        return []

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.10:
            results.append(loaded_chunks[idx])

    return results

@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(semantic_search, user_query, 3)

            if not matched_results:
                yield {"data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."}
                return

            extracted_text = "\n\n---\n\n".join(matched_results)
            words = extracted_text.split(" ")
            
            for i in range(0, len(words), 2):
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.01)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
