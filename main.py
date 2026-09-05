import asyncio
from contextlib import asynccontextmanager
import json
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# تحديد مجلد العمل ومسار ملف JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "output_data.json")

loaded_data: List[Dict[str, Any]] = []


def clean_text(text: str) -> str:
    """تنظيف النص لتسهيل المطابقة (إزالة التشكيل والهمزات)"""
    if not text:
        return ""
    text = text.lower()
    # إزالة التشكيل العربي
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # توحيد الهمزات والأحرف
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    return text.strip()


def load_data():
    """تحميل ملف output_data.json في الذاكرة عند بدء التشغيل"""
    global loaded_data

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"❌ لم يتم العثور على الملف: {DATA_FILE}. يرجى التأكد من وجود output_data.json في المجلد."
        )

    print("⚡ جاري تحميل بيانات output_data.json ...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = json.load(f)
        # التعامل مع الملف سواء كان مصفوفة مباشرة أو كائن يحتوي على حقل "data"
        if isinstance(content, list):
            loaded_data = content
        elif isinstance(content, dict) and "data" in content:
            loaded_data = content["data"]
        else:
            loaded_data = []

    print(f"✅ تم تحميل البيانات بنجاح: {len(loaded_data)} عنصر/سؤال.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(title="Nineveh Edu Search - Local Search", lifespan=lifespan)

# إعداد CORS للاتصال من أي واجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


def keyword_search(query: str, top_k: int = 1) -> List[Dict[str, Any]]:
    """دالة البحث النصي المرنة مع حساب النقاط وتخفيض العتبة"""
    if not loaded_data or not query.strip():
        return []

    cleaned_query = clean_text(query)
    query_words = [w for w in cleaned_query.split() if len(w) > 1]

    # أداء تصفية الكلمات الزائدة/الأدوات
    stopwords = {
        "عن",
        "في",
        "على",
        "من",
        "إلى",
        "ما",
        "هي",
        "هو",
        "كم",
        "كيف",
        "هل",
        "متى",
        "منهو",
        "شنو",
        "ماهي",
        "ماهو",
    }
    filtered_words = [w for w in query_words if w not in stopwords]

    # إذا كانت كل الكلمات أدوات، نستخدم الكلمات الأصلية
    search_words = filtered_words if filtered_words else query_words

    results = []

    for item in loaded_data:
        score = 0
        question_clean = clean_text(item.get("question", ""))
        answer_clean = clean_text(item.get("answer", ""))

        raw_keywords = item.get("keywords", [])
        keywords_clean = [
            clean_text(str(k)) for k in raw_keywords if isinstance(k, (str, int))
        ]

        # 1. مطابقة عبارة البحث بالكامل داخل السؤال
        if cleaned_query in question_clean:
            score += 10

        # 2. مطابقة الكلمات المنفردة
        for word in search_words:
            if word in question_clean:
                score += 3  # تطابق في نص السؤال
            if any(word in kw for kw in keywords_clean):
                score += 2  # تطابق في الكلمات المفتاحية
            if word in answer_clean:
                score += 1  # تطابق في نص الإجابة

        if score > 0:
            results.append({"score": score, "item": item})

    # ترتيب النتائج من الأكبر نقاطاً للأقل
    results.sort(key=lambda x: x["score"], reverse=True)

    # إرجاع أعلى نتيجة إذا حققت شرط الاستجابة (حتى لو كانت نقطة واحدة)
    if results and results[0]["score"] >= 1:
        return [results[0]["item"]]

    return []


@app.get("/")
def read_root():
    return {"status": "online", "loaded_records": len(loaded_data)}


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(
                keyword_search, user_query, 1
            )

            if not matched_results:
                yield {
                    "data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في دليل التعليمات المتاح."
                }
                return

            best_match = matched_results[0]
            answer_text = best_match.get("answer", "")

            if not answer_text:
                yield {"data": "لم يتم العثور على نص الإجابة لهذا السؤال."}
                return

            # بث النص كلمة كلمة لتجربة تفاعلية (Streaming)
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
