import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import List

import cohere
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")

# قراءة API Key الخاص بـ Cohere من متغيرات البيئة
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# تهيئة عميل Cohere
co_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None


def get_query_embedding_from_cohere(text: str) -> np.ndarray:
    """استخراج متجه النص باستخدام نموذج Cohere متعدد اللغات v3.0."""
    if not co_client:
        raise Exception("❌ لم يتم العثور على COHERE_API_KEY في متغيرات البيئة!")

    try:
        # استخدام input_type="search_query" المخصص للاستعلامات في Cohere
        response = co_client.embed(
            texts=[text],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        )

        embeddings = np.array(response.embeddings, dtype=np.float32)

        # حساب L2 Normalization لتسهيل عملية المقارنة
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    except Exception as e:
        print(f"❌ خطأ أئناء الاتصال بـ Cohere API: {str(e)}")
        raise Exception(f"فشل الحصول على متجهات البحث من Cohere: {str(e)}")


def load_data():
    global loaded_chunks, chunk_embeddings

    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("❌ ملفات المتجهات غير موجودة! يرجى التأكد من رفع embeddings.npy و chunks.json.")

    print("⚡ جاري تحميل ملفات المتجهات والنصوص...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print(f"✅ تم تحميل البيانات بنجاح بأبعاد: {chunk_embeddings.shape}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(title="Nineveh Edu Search - Cohere API Backend", lifespan=lifespan)

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
    if chunk_embeddings is None or len(loaded_chunks) == 0:
        return []

    # 1. استخراج متجه البحث من Cohere
    query_vec = get_query_embedding_from_cohere(query)[0]

    # 2. التأكد من توافق أبعاد المتجهات
    if chunk_embeddings.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"⚠️ عدم تطابق الأبعاد! أبعاد الملف الحالي هي {chunk_embeddings.shape[1]} بينما متجه Cohere هو {query_vec.shape[0]}. "
            "يرجى إعادة توليد ملف embeddings.npy باستخدام Cohere."
        )

    # 3. حساب التشابه بالضرب النقطي (Dot Product)
    similarities = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.15:  # عتبة التشابه
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
