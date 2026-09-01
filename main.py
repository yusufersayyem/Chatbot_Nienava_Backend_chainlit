import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

import requests
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ==========================================
# 1. الإعدادات والتهيئات العامة
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")
JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]

# مفتاح Hugging Face يُقرأ من متغيرات البيئة تلقائياً
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "ضع_رمز_HF_هنا_إن_لم_تستخدم_متغيرات_البيئة")

# الرابط الرسمي لخدمة Hugging Face Inference Router
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None

# ==========================================
# 2. دالة الاتصال بـ Hugging Face مع حماية 429
# ==========================================
def get_embeddings_from_hf(texts: List[str]) -> np.ndarray:
    """إرسال النصوص إلى Hugging Face وتوليد المتجهات مع معالجة تجاوز حدود الطلبات (Rate Limits)."""
    max_retries = 5
    retry_delay = 5  # الانتظار 5 ثوانٍ عند حدوث 429

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30
            )

            if response.status_code == 200:
                emb_data = response.json()
                embeddings = np.array(emb_data, dtype=np.float32)
                
                # تطبيق Normalization لتسهيل حساب Cosine / Dot Product
                if embeddings.ndim == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)
                
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                return embeddings / np.maximum(norms, 1e-12)

            elif response.status_code == 429:
                print(f"⚠️ تجاوز الحد المسموح (429). جاري الانتظار {retry_delay} ثوانٍ... (محاولة {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # زيادة وقت الانتظار تصاعدياً

            elif response.status_code == 503:
                print(f"⏳ النموذج يتم تحميله على HF (503). انتظار 10 ثوانٍ...")
                time.sleep(10)

            else:
                raise Exception(f"خطأ HF API ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as req_err:
            print(f"⚠️ خطأ في الاتصال بالحزم: {req_err}. إعادة المحاولة...")
            time.sleep(3)

    raise Exception("❌ فشل استخراج المتجهات من Hugging Face بعد عدة محاولات بسبب كثرة الطلبات (429).")

# ==========================================
# 3. دورة حياة التطبيق وتحميل البيانات
# ==========================================
async def prepare_and_embed_data():
    """تحميل ملفات JSON واستخراج متجهاتها عند تشغيل السيرفر."""
    global loaded_chunks, chunk_embeddings

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
                            formatted_str = "\n".join([f"{k}: {v}" for k, v in item.items()])
                        else:
                            formatted_str = str(item)
                        raw_texts.append(formatted_str)
            except Exception as e:
                print(f"❌ خطأ أثناء قراءة {file_name}: {e}")

    loaded_chunks = raw_texts

    if loaded_chunks:
        print(f"⏳ جاري توليد المتجهات لـ ({len(loaded_chunks)}) نص عبر API...")
        try:
            # إرسال النصوص في دفعات صغيرة (Batch Size = 16) لتفادي حدود HF
            batch_size = 16
            all_embeddings = []
            for i in range(0, len(loaded_chunks), batch_size):
                batch = loaded_chunks[i : i + batch_size]
                batch_emb = await asyncio.to_thread(get_embeddings_from_hf, batch)
                all_embeddings.append(batch_emb)
                await asyncio.sleep(0.5)  # مهلة زمنية قصيرة بين الدفعات

            chunk_embeddings = np.vstack(all_embeddings)
            print("✅ تم استخراج وتخزين المتجهات بنجاح في الذاكرة!")
        except Exception as e:
            print(f"❌ خطأ أثناء استخراج المتجهات: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(prepare_and_embed_data())
    yield

app = FastAPI(title="HuggingFace API Search - Nineveh Edu", lifespan=lifespan)

# تفعيل CORS للاتصال السلس مع الفرونت إند
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
# 4. دالة البحث الدلالي السريعة
# ==========================================
def semantic_search(query: str, top_k: int = 3) -> List[str]:
    if chunk_embeddings is None or len(loaded_chunks) == 0:
        return []

    # استخراج متجه الاستعلام من API
    query_vec = get_embeddings_from_hf([query])[0]

    # حساب نسبة التشابه الدلالي (Dot Product)
    similarities = np.dot(chunk_embeddings, query_vec)

    # ترتيب أفضل النتائج
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.15:
            results.append(loaded_chunks[idx])

    return results

# ==========================================
# 5. Endpoint بث النتائج (Streaming)
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

            # بث الكلمات تدريجياً لسرعة العرض في Chainlit
            words = extracted_text.split(" ")
            for i in range(0, len(words), 2):
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.01)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
