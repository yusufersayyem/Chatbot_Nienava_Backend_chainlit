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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "output_data.json")

loaded_data: List[Dict[str, Any]] = []


def clean_text(text: str) -> str:
    """تنظيف النص لتسهيل عملية المطابقة"""
    text = text.lower()
    # إزالة التشكيل والهمزات للتطابق المرن
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    return text


def load_data():
    """تحميل ملف output_data.json في الذاكرة عند بدء التشغيل"""
    global loaded_data

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"❌ لم يتم العثور على الملف: {DATA_FILE}. يرجى التأكد من رفع output_data.json إلى المجلد الرئيسي."
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

    print(f"✅ تم تحميل البيانات بنجاح: {len(loaded_data)} سؤال وجواب.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield


app = FastAPI(title="Nineveh Edu Search - Local Keyword Search", lifespan=lifespan)

# إعداد CORS
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
    """البحث النصي عن طريق المطابقة وحساب النقاط (Scoring System) بدون ذكاء اصطناعي"""
    if not loaded_data or not query.strip():
        return []

    cleaned_query = clean_text(query)
    query_words = [w for w in cleaned_query.split() if len(w) > 2]

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
    }
    filtered_query_words = [w for w in query_words if w not in stopwords]

    results = []

    for item in loaded_data:
        score = 0

        question_clean = clean_text(item.get("question", ""))
        answer_clean = clean_text(item.get("answer", ""))
        keywords_clean = [
            clean_text(k) for k in item.get("keywords", []) if isinstance(k, str)
        ]

        # 1. مطابقة الجملة الكاملة في السؤال (تأخذ أعلى وزن)
        if cleaned_query in question_clean:
            score += 10

        # 2. مطابقة الكلمات المنفردة
        for word in filtered_query_words:
            if word in question_clean:
                score += 4  # مطابقة في السؤال
            if any(word in kw for kw in keywords_clean):
                score += 3  # مطابقة في الكلمات المفتاحية
            if word in answer_clean:
                score += 1  # مطابقة في الإجابة

        if score > 0:
            results.append({"score": score, "item": item})

    # ترتيب النتائج بناءً على النقاط من الأعلى للأقل
    results.sort(key=lambda x: x["score"], reverse=True)

    # إرجاع أفضل النتائج التي حققت حداً أدنى من النقاط
    return [r["item"] for r in results[:top_k] if r["score"] >= 3]


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

            # بث النص كلمة كلمة لتجربة مستخدم تفاعلية (Streaming)
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
