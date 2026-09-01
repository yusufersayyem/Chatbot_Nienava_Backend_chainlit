import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
from typing import List

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sse_starlette.sse import EventSourceResponse

# ==========================================
# 1. الإعدادات والتهيئات
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")
JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]

# تحميل النموذج محلياً (يتم تحميله لمرة واحدة عند التشغيل)
# نموذج bge-m3 أو نموذج أسرع وأخف للغة العربية مثل sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
model = None

loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None


async def prepare_and_embed_data():
    """تحميل النصوص وحساب المتجهات عند بدء الخادم."""
    global loaded_chunks, chunk_embeddings, model

    print("⏳ جاري تحميل نموذج التضمين...")
    # تحميل النموذج في الذاكرة
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ تم تحميل النموذج بنجاح.")

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
            except Exception as e:
                print(f"❌ خطأ أثناء قراءة {file_name}: {e}")

    loaded_chunks = raw_texts
    if loaded_chunks:
        print(f"⏳ جاري توليد المتجهات لـ ({len(loaded_chunks)}) نص...")
        # encode تحسب جميع المتجهات دفعة واحدة بسرعة فائقة
        embeddings = await asyncio.to_thread(
            model.encode, loaded_chunks, normalize_embeddings=True
        )
        chunk_embeddings = np.array(embeddings)
        print("✅ اكتمل استخراج المتجهات بنجاح.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(prepare_and_embed_data())
    yield


app = FastAPI(
    title="HuggingFace BGE-M3 API Search - Nineveh Edu", lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str


# ==========================================
# 2. دالة البحث الدلالي السريعة
# ==========================================
def semantic_search(query: str, top_k: int = 3) -> List[str]:
    if chunk_embeddings is None or len(loaded_chunks) == 0 or model is None:
        return []

    # حساب متجه الاستعلام محلياً وبثوانٍ معدودة
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    # حساب Dot Product (لأن المتجهات مُطَبّقة Normalize)
    similarities = np.dot(chunk_embeddings, query_vec)

    # ترتيب النتائج
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # خفضنا العتبة إلى 0.15 لتجنب رفض الإجابات القريبة دلالياً
        if similarities[idx] > 0.15:
            results.append(loaded_chunks[idx])

    return results


# ==========================================
# 3. Endpoints
# ==========================================
@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    # التحيات المباشرة... (نفس الكود السابق)

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(
                semantic_search, user_query, 3
            )

            if not matched_results:
                yield {
                    "data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."
                }
                return

            extracted_text = "\n\n---\n\n".join(matched_results)

            # إرسال النص بسرعة مناسبة للفرونت إند
            words = extracted_text.split(" ")
            for i in range(0, len(words), 2):  # إرسال كلمتين في كل مرة لسرعة أسرع
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.01)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
