import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import fuzz, process
from sse_starlette.sse import EventSourceResponse

# تحديد مسار الملفات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "output_data.json")

loaded_data: List[Dict[str, Any]] = []


def clean_text(text: str) -> str:
    """تنظيف النص وتوحيد الأحرف العربية"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[أإآ]", "ا", text)  # توحيد الهمزات
    text = re.sub(r"ة", "ه", text)  # توحيد التاء المربوطة
    text = re.sub(r"ى", "ي", text)  # توحيد الألف المقصورة
    return text.strip()


def load_data():
    """تحميل ملف output_data.json عند التشغيل"""
    global loaded_data

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ لم يتم العثور على الملف: {DATA_FILE}")

    print("⚡ جاري تحميل ملف الأسئلة والأجوبة...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = json.load(f)
        if isinstance(content, list):
            loaded_data = content
        elif isinstance(content, dict) and "data" in content:
            loaded_data = content["data"]
        else:
            loaded_data = []

    print(f"✅ تم تحميل {len(loaded_data)} سؤال وجواب بنجاح.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(
    title="Nineveh Edu Search - Rapid Fuzzy Search", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def smart_fuzzy_search(query: str) -> Dict[str, Any]:
    """محرك البحث الذكي بالمطابقة الضبابية والكلمات المفتاحية"""
    if not loaded_data or not query.strip():
        return None

    cleaned_query = clean_text(query)

    best_match = None
    highest_score = 0.0

    for item in loaded_data:
        question_text = clean_text(item.get("question", ""))
        answer_text = clean_text(item.get("answer", ""))

        # جلب الكلمات المفتاحية وتحويلها لنص موحد
        raw_keywords = item.get("keywords", [])
        keywords_str = " ".join([clean_text(str(k)) for k in raw_keywords])

        # 1. حساب نسبة تشابه السؤال المطروح مع السؤال المخزن (Partial & Token Ratio)
        q_ratio = fuzz.token_set_ratio(cleaned_query, question_text)

        # 2. حساب نسبة التشابه مع الكلمات المفتاحية
        kw_ratio = fuzz.partial_ratio(cleaned_query, keywords_str)

        # 3. حساب نسبة التشابه مع نص الإجابة نفسها
        ans_ratio = fuzz.partial_ratio(cleaned_query, answer_text)

        # معادلة وزن النقاط الإجمالية
        total_score = (q_ratio * 0.6) + (kw_ratio * 0.3) + (ans_ratio * 0.1)

        # إذا تطابقت كلمة مفتاحية رئيسية بالكامل يُمنح بونص إضافي
        for word in cleaned_query.split():
            if len(word) > 2 and word in keywords_str:
                total_score += 15

        if total_score > highest_score:
            highest_score = total_score
            best_match = item

    # طباعة أعلى نسبة تشابه للـ Debug في سيرفر Render
    print(
        f"🔍 أفضل مطابقة للاستعلام [{query}]: النسبة = {highest_score:.2f}%"
    )

    # قبول الإجابة إذا تجاوزت نسبة التشابه 35%
    if best_match and highest_score >= 35.0:
        return best_match

    return None


@app.get("/")
def read_root():
    return {"status": "online", "loaded_records": len(loaded_data)}


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    async def json_generator():
        try:
            matched_item = await asyncio.to_thread(
                smart_fuzzy_search, user_query
            )

            if not matched_item:
                yield {
                    "data": "عذراً، لم أجد إجابة مطابقة لاستفسارك في دليل التعليمات المتاح. يرجى كتابة كلمات مفتاحية أوردها الدليل."
                }
                return

            answer_text = matched_item.get("answer", "")

            if not answer_text:
                yield {"data": "لم يتم العثور على نص الإجابة لطلبك."}
                return

            # إرسال الإجابة بأسلوب Streaming تفاعلي
            words = answer_text.split(" ")
            for i in range(0, len(words), 2):
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.02)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
