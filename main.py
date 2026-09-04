import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import List, Dict, Any

import cohere
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")

# قراءة مفتاح Cohere من متغيرات بيئة Render
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# تهيئة عميل Cohere الذكي
co_client = None
if COHERE_API_KEY:
    try:
        co_client = cohere.ClientV2(COHERE_API_KEY)
    except AttributeError:
        co_client = cohere.Client(COHERE_API_KEY)

loaded_chunks: List[Dict[str, Any]] = []
chunk_embeddings: np.ndarray = None


def get_query_embedding_from_cohere(text: str) -> np.ndarray:
    """استخراج متجه النص باستخدام نموذج Cohere متعدد اللغات v3.0 مع L2 Normalization."""
    if not co_client:
        raise Exception("❌ لم يتم العثور على COHERE_API_KEY في متغيرات البيئة!")

    try:
        response = co_client.embed(
            texts=[text],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        )

        # استخراج المتجهات بغض النظر عن إصدار المكتبة
        if hasattr(response, "embeddings"):
            if hasattr(response.embeddings, "float"):
                emb_list = response.embeddings.float
            else:
                emb_list = response.embeddings
        else:
            emb_list = response

        embeddings = np.array(emb_list, dtype=np.float32)

        # حساب L2 Normalization لتسهيل مقارنة الضرب النقطي
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    except Exception as e:
        print(f"❌ خطأ أثناء الاتصال بـ Cohere API: {str(e)}")
        raise Exception(f"فشل الحصول على متجهات البحث من Cohere: {str(e)}")


def load_data():
    """تحميل الملفات في الذاكرة عند بدء تشغيل الخادم."""
    global loaded_chunks, chunk_embeddings

    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(
            "❌ ملفات المتجهات غير موجودة! يرجى التأكد من رفع embeddings.npy و chunks.json إلى المجلد الرئيسي."
        )

    print("⚡ جاري تحميل ملفات المتجهات والنصوص الـ 199...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print(f"✅ تم تحميل البيانات بنجاح: {len(loaded_chunks)} كتلة بأبعاد {chunk_embeddings.shape}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(title="Nineveh Edu Search - Cohere API Backend", lifespan=lifespan)

# إعداد CORS للعمل مع أي واجهة أمامية متصلة بـ Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def semantic_search(query: str, top_k: int = 1) -> List[Dict[str, Any]]:
    """البحث الدلالي لحساب تشابه الضرب النقطي وإرجاع أفضل الكتل."""
    if chunk_embeddings is None or len(loaded_chunks) == 0:
        return []

    # 1. استخراج متجه البحث
    query_vec = get_query_embedding_from_cohere(query)[0]

    # 2. التأكد من توافق الأبعاد
    if chunk_embeddings.shape[1] != query_vec.shape[0]:
        raise ValueError(
            f"⚠️ عدم تطابق الأبعاد! الملف يحوي {chunk_embeddings.shape[1]} بينما المتجه يحوي {query_vec.shape[0]}."
        )

    # 3. حساب الضرب النقطي
    similarities = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # عتبة التشابه المناسبة للغة العربية
        if similarities[idx] > 0.20:
            results.append(loaded_chunks[idx])

    return results


@app.get("/")
def read_root():
    """مسار اختبار صحة الخادم (Health Check) لمنصة Render."""
    return {"status": "online", "loaded_chunks": len(loaded_chunks)}


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(semantic_search, user_query, 1)

            if not matched_results:
                yield {"data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."}
                return

            # استخراج الجواب المباشر من الكائنات الهيكلية
            best_match = matched_results[0]
            answer_text = best_match.get("answer", best_match.get("text", ""))

            # تقسيم الجواب إلى كلمات لتدفق البيانات (Streaming)
            words = answer_text.split(" ")

            for i in range(0, len(words), 2):
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.03)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())


if __name__ == "__main__":
    import uvicorn
    # ربط المنفذ بمتغير البيئة PORT الخاص بـ Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
