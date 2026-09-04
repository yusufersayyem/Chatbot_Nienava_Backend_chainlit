import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import requests
from sse_starlette.sse import EventSourceResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")

# رابط الـ Inference API المباشر لـ Hugging Face
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# قراءة HF_TOKEN من متغيرات بيئة Render
HF_TOKEN = os.getenv("HF_TOKEN", "")

# إعداد الرأسيات للتأكد من تمرير التوكين وتقليل احتمال 429
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
} if HF_TOKEN else {"Content-Type": "application/json"}

loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None


def get_query_embedding_from_hf(text: str) -> np.ndarray:
    """استخراج متجه النص عبر API مع معالجة حظر 429 وإعادة المحاولة التلقائية."""
    max_retries = 5
    retry_delay = 2.0  # التمهل الابتدائي بالثواني

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": [text], "options": {"wait_for_model": True}},
                timeout=25
            )

            if response.status_code == 200:
                emb_data = response.json()
                embeddings = np.array(emb_data, dtype=np.float32)

                if embeddings.ndim == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)

                # L2 Normalization
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                return embeddings / np.maximum(norms, 1e-12)

            # معالجة 429 (Rate Limit) و 503 (Model Loading) عبر الانتظار التصاعدي
            elif response.status_code in (429, 503):
                print(f"⚠️ تنبيه من HF API ({response.status_code}). محاولة {attempt + 1}/{max_retries} - انتظار {retry_delay} ثوانٍ...")
                time.sleep(retry_delay)
                retry_delay *= 1.8  # Exponential Backoff
            else:
                raise Exception(f"HF API Error ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ خطأ اتصال: {e}. محاولة جديدة خلال ثانيتين...")
            time.sleep(2)

    raise Exception("تجاوز الحد المسموح للطلبات (429). يرجى التأكد من ضبط HF_TOKEN أو الانتظار لحظات.")


def load_data():
    global loaded_chunks, chunk_embeddings

    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("❌ ملفات المتجهات غير موجودة! يرجى رفع embeddings.npy و chunks.json.")

    print("⚡ جاري تحميل المتجهات والنصوص الجاهزة...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print("✅ الباك إند جاهز ويعمل عبر الـ API بنجاح دون استهلاك ذاكرة RAM!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(title="Nineveh Edu Search - HF API Backend", lifespan=lifespan)

# إعدادات CORS للسماح بالاتصال من أي فرونت إند
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

    # استخراج متجه النص عبر Hugging Face API
    query_vec = get_query_embedding_from_hf(query)[0]

    # حساب تشابه ضرب المتجهات (Dot Product)
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
