import asyncio
import json
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Normalized JSON Search Backend - Nineveh Edu")

# ==========================================
# 1. تحديد مسارات الملفات وتحميل البيانات
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")

JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]
loaded_data: List[Dict[str, Any]] = []


def normalize_arabic(text: str) -> str:
    """إزالة التشكيل والتطويل وتوحيد الهمزات والأحرف العربية."""
    if not text:
        return ""
    # إزالة حركات التشكيل والتنوين
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0653-\u065F\u0670]", "", text)
    # إزالة التطويل (الكشيدة)
    text = re.sub(r"\u0640", "", text)
    # توحيد الألفات والهمزات
    text = re.sub(r"[أإآ]", "ا", text)
    # توحيد الياء والتاء المربوطة
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text.lower().strip()


def load_all_json_files():
    """تحميل كافة ملفات JSON إلى الذاكرة لضمان السرعة الأقصى."""
    global loaded_data
    loaded_data = []

    for file_name in JSON_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    items = content if isinstance(content, list) else [content]
                    for item in items:
                        loaded_data.append({"file": file_name, "content": item})
                print(f"✅ تم تحميل الملف بنجاح: {file_name}")
            except Exception as e:
                print(f"❌ خطأ أثناء قراءة {file_name}: {e}")
        else:
            print(f"⚠️ الملف غير موجود -> {file_path}")


# تشغيل التحميل عند الإقلاع
load_all_json_files()

# ==========================================
# 2. خريطة التحيات
# ==========================================
GREETINGS_MAP = {
    r"^(مرحبا|مرحباً|أاهلا|أهلاً|اهلين|أهلين|السلام عليكم|مرحبتين|هلا|صباح الخير|مساء الخير)": (
        "أهلاً بك! أنا المساعد الذكي للمديرية العامة لتربية نينوى. كيف"
        " يمكنني مساعدتك اليوم؟"
    ),
    r"^(من انت|من أنت|عرف عن نفسك|ما هو عملك|ماذا تفعل)": (
        "أنا مساعد مخصص للبحث في المستندات والتعليمات الرسمية للمديرية"
        " العامة لتربية نينوى."
    ),
    r"^(شكرا|شكراً|يعطيك العافية|تسلم|تسلم ايدك|مشكور)": (
        "العفو! أنا في الخدمة دائماً. هل لديك أي استفسار إداري آخر؟"
    ),
}


class QueryRequest(BaseModel):
    query: str


# ==========================================
# 3. دالة البحث النصي ذكية الأوزان
# ==========================================
def search_in_json(query: str) -> List[str]:
    results = []
    norm_query = normalize_arabic(query)
    keywords = [k for k in norm_query.split() if len(k) > 1]

    if not keywords:
        return results

    for record in loaded_data:
        raw_text = json.dumps(record["content"], ensure_ascii=False)
        norm_text = normalize_arabic(raw_text)

        score = 0
        for kw in keywords:
            # مطابقة تامة للكلمة = 2 نقطة
            if re.search(r"\b" + re.escape(kw) + r"\b", norm_text):
                score += 2
            # مطابقة جزئية = 1 نقطة
            elif kw in norm_text:
                score += 1

        if score > 0:
            if isinstance(record["content"], dict):
                formatted = "\n".join([f"**{k}**: {v}" for k, v in record["content"].items()])
            else:
                formatted = str(record["content"])
            results.append((score, formatted))

    # ترتيب النتائج بالأعلى نقاطاً
    results.sort(key=lambda x: x[0], reverse=True)
    return [res[1] for res in results[:5]]


# ==========================================
# 4. نقاط النهاية (Endpoints)
# ==========================================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "loaded_records": len(loaded_data),
        "data_directory": DATA_DIR,
    }


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    # أولاً: الرد السريع للتحيات
    for pattern, response_text in GREETINGS_MAP.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            async def greeting_generator():
                for word in response_text.split(" "):
                    yield {"data": word + " "}
                    await asyncio.sleep(0.01)
            return EventSourceResponse(greeting_generator())

    # ثانياً: البحث والبث
    async def json_generator():
        try:
            if not loaded_data:
                yield {"data": "عذراً، لم يتم تحميل أي بيانات في النظام."}
                return

            matched_results = await asyncio.to_thread(search_in_json, user_query)

            if not matched_results:
                yield {"data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."}
                return

            extracted_text = "\n\n---\n\n".join(matched_results)

            for word in extracted_text.split(" "):
                yield {"data": word + " "}
                await asyncio.sleep(0.005)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
