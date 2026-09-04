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

# ==========================================
# 1. المسارات والتهيئات
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# المتغيرات العامة في الذاكرة
model: SentenceTransformer = None
loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None

# ==========================================
# 2. تحميل البيانات الجاهزة فقط عند الإقلاع
# ==========================================
def load_prepared_data():
    global model, loaded_chunks, chunk_embeddings

    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(
            "❌ ملفات المتجهات غير موجودة! يرجى تشغيل سكريبت `embedder.py` أولاً لتوليدها."
        )

    print("⚡ جاري تحميل المتجهات والنصوص الجاهزة من القرص...")
    # تحميل المتجهات والنصوص فوراً من الملفات المحفوظة
    chunk_embeddings = np.load(EMBEDDINGS_FILE)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print("⏳ جاري تحميل نموذج التضمين للاستعلامات...")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ الباك إند جاهز للعمل بكفاءة عالية!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # تحميل الملفات في مسار غير حاصر (Non-blocking)
    await asyncio.to_thread(load_prepared_data)
    yield

app = FastAPI(title="Fast Numpy Semantic Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# ==========================================
# 3. دالة البحث الدلالي
# ==========================================
def semantic_search(query: str, top_k: int = 3) -> List[str]:
    if model is None or chunk_embeddings is None or len(loaded_chunks) == 0:
        return []

    # استخراج متجه البحث للسؤال فقط
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    # حساب ضرب التشابه (Dot Product) فورياً عبر NumPy
    similarities = np.dot(chunk_embeddings, query_vec)

    # ترتيب النتائج من الأعلى للأقل
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.10:
            results.append(loaded_chunks[idx])

    return results

# ==========================================
# 4. Endpoint البث (Streaming)
# ==========================================
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
