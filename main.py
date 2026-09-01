import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from huggingface_hub import InferenceClient
import numpy as np
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ==========================================
# 1. الإعدادات والتهيئات
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")
JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]

hf_client = InferenceClient(model=EMBEDDING_MODEL_ID, token=HF_TOKEN)

loaded_chunks: List[str] = []
chunk_embeddings: List[List[float]] = []


def get_embedding_via_api(text: str) -> List[float]:
    """استدعاء Hugging Face API للحصول على Vector النص."""
    try:
        response = hf_client.feature_extraction(text)
        if hasattr(response, "tolist"):
            response = response.tolist()

        while (
            isinstance(response, list)
            and len(response) > 0
            and isinstance(response[0], list)
        ):
            if isinstance(response[0][0], list):
                response = response[0]
            else:
                break

        if (
            isinstance(response, list)
            and len(response) > 0
            and isinstance(response[0], list)
        ):
            response = [sum(col) / len(response) for col in zip(*response)]

        return response
    except Exception as e:
        print(f"❌ خطأ في الحصول على التضمين من API للمتن: {e}")
        return []


async def prepare_and_embed_data():
    """تحميل النصوص وحساب المتجهات بشكل غير متزامني (Async) عند بدء التطبيق."""
    global loaded_chunks, chunk_embeddings
    loaded_chunks = []
    chunk_embeddings = []

    raw_texts = []
    for file_name in JSON_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    items = content if isinstance(content, list) else [content]
                    for item in items:
                        if isinstance(item, dict):
                            formatted_str = "\n".join(
                                [f"{k}: {v}" for k, v in item.items()]
                            )
                        else:
                            formatted_str = str(item)
                        raw_texts.append(formatted_str)
                print(f"✅ تم تحميل النصوص من: {file_name}")
            except Exception as e:
                print(f"❌ خطأ أثناء قراءة {file_name}: {e}")

    loaded_chunks = raw_texts
    if loaded_chunks:
        print(
            f"⏳ جاري استخراج Embeddings لـ ({len(loaded_chunks)}) نص عبر Hugging Face API..."
        )
        for i, text in enumerate(loaded_chunks):
            # تشغيل الدالة الإجرائية داخل Thread لتجنب حظر السيرفر
            emb = await asyncio.to_thread(get_embedding_via_api, text)
            if emb:
                chunk_embeddings.append(emb)
            else:
                chunk_embeddings.append([0.0] * 1024)

            # استخدام asyncio.sleep بشكل صحيح
            await asyncio.sleep(0.05)

        print("✅ اكتمل استخراج المتجهات من API بنجاح.")


# إدارة دورة حياة التطبيق (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # يُنفذ عند بدء تشغيل الخادم
    asyncio.create_task(prepare_and_embed_data())
    yield
    # يُنفذ عند إغلاق الخادم


app = FastAPI(
    title="HuggingFace BGE-M3 API Search - Nineveh Edu", lifespan=lifespan
)

# ==========================================
# 2. خريطة التحيات المباشرة
# ==========================================
GREETINGS_MAP = {
    r"^(مرحبا|مرحباً|أاهلا|أهلاً|اهلين|أهلين|السلام عليكم|مرحبتين|هلا|صباح الخير|مساء الخير)": (
        "أهلاً بك! أنا المساعد الذكي للمديرية العامة لتربية نينوى. كيف يمكنني مساعدتك اليوم؟"
    ),
    r"^(من انت|من أنت|عرف عن نفسك|ما هو عملك|ماذا تفعل)": (
        "أنا مساعد مخصص للبحث في المستندات والتعليمات الرسمية للمديرية العامة لتربية نينوى."
    ),
    r"^(شكرا|شكراً|يعطيك العافية|تسلم|تسلم ايدك|مشكور)": (
        "العفو! أنا في الخدمة دائماً. هل لديك أي استفسار إداري آخر؟"
    ),
}


class QueryRequest(BaseModel):
    query: str


# ==========================================
# 3. دالة البحث الدلالي باستخدام API التشابه
# ==========================================
def semantic_search(query: str, top_k: int = 5) -> List[str]:
    if not chunk_embeddings or len(loaded_chunks) == 0:
        return []

    query_vec = get_embedding_via_api(query)
    if not query_vec:
        return []

    query_np = np.array(query_vec)
    chunks_np = np.array(chunk_embeddings)

    query_norm = query_np / (np.linalg.norm(query_np) + 1e-10)
    chunks_norm = chunks_np / (
        np.linalg.norm(chunks_np, axis=1, keepdims=True) + 1e-10
    )

    similarities = np.dot(chunks_norm, query_norm)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.30:
            results.append(loaded_chunks[idx])

    return results


# ==========================================
# 4. نقاط النهاية (Endpoints)
# ==========================================
@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "loaded_records": len(loaded_chunks),
        "processed_embeddings": len(chunk_embeddings),
        "embedding_model": EMBEDDING_MODEL_ID,
        "mode": "Hugging Face Inference API",
    }


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    for pattern, response_text in GREETINGS_MAP.items():
        if re.search(pattern, user_query, re.IGNORECASE):

            async def greeting_generator():
                for word in response_text.split(" "):
                    yield {"data": word + " "}
                    await asyncio.sleep(0.01)

            return EventSourceResponse(greeting_generator())

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(
                semantic_search, user_query, 5
            )

            if not matched_results:
                yield {
                    "data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."
                }
                return

            extracted_text = "\n\n---\n\n".join(matched_results)

            for word in extracted_text.split(" "):
                yield {"data": word + " "}
                await asyncio.sleep(0.005)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
