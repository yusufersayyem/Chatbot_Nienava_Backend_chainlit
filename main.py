import asyncio
import json
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="JSON Search Backend - Nineveh Edu")

# ==========================================
# 1. تحديد مسارات الملفات وتفريغها عند التشغيل
# ==========================================
# تحديد مسار المجلد الذي يحتوي على ملف main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحديد المسار المطلق للمجلد الفرعي الذي يحتوي على ملفات JSON
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")

JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]
loaded_data: List[Dict[str, Any]] = []


def load_all_json_files():
    """تحميل كافة ملفات JSON من مجلد loaded_data إلى الذاكرة لضمان سرعة الاستجابة."""
    global loaded_data
    loaded_data = []

    for file_name in JSON_FILES:
        # بناء المسار الكامل للملف داخل مجلد loaded_data
        file_path = os.path.join(DATA_DIR, file_name)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    # دعم كل من المصفوفات (List) والكائنات (Dict)
                    if isinstance(content, list):
                        for item in content:
                            loaded_data.append({"file": file_name, "content": item})
                    else:
                        loaded_data.append({"file": file_name, "content": content})
                print(f"✅ تم تحميل الملف بنجاح: {file_name}")
            except Exception as e:
                print(f"❌ خطأ أثناء قراءة الملف {file_name}: {e}")
        else:
            print(f"⚠️ الملف غير موجود في المسار -> {file_path}")


# استدعاء دالة التحميل فور تشغيل الخادم
load_all_json_files()

# ==========================================
# 2. خريطة التحيات المباشرة
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
# 3. دالة البحث النصي داخل بيانات JSON
# ==========================================
def search_in_json(query: str) -> List[str]:
    """تقتفي أثر الكلمات المفتاحية في كل كائن/مستند داخل بيانات JSON."""
    results = []
    keywords = [k.lower() for k in query.split() if len(k) > 1]

    if not keywords:
        return results

    for record in loaded_data:
        # تحويل محتوى الـ JSON بالكامل إلى نص واحد لإجراء البحث المباشر
        text_content = json.dumps(record["content"], ensure_ascii=False)
        text_lower = text_content.lower()

        # حساب عدد الكلمات المفتاحية المطابقة
        matches = sum(1 for kw in keywords if kw in text_lower)

        if matches > 0:
            # إذا كان العنصر عبارة عن كائن يحتوي على حقول نصية، استخرج النص بشكل منسق
            if isinstance(record["content"], dict):
                formatted_item = "\n".join(
                    [f"**{k}**: {v}" for k, v in record["content"].items()]
                )
            else:
                formatted_item = str(record["content"])

            results.append((matches, formatted_item))

    # ترتيب النتائج بناءً على الأكثر مطابقة للكلمات المفتاحية
    results.sort(key=lambda x: x[0], reverse=True)

    # إرجاع أعلى 5 نتائج فقط لتجنب الإطالة
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
        "message": "Server is active",
    }


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    # أولاً: الرد السريع المباشر للتحيات
    for pattern, response_text in GREETINGS_MAP.items():
        if re.search(pattern, user_query, re.IGNORECASE):

            async def greeting_generator():
                for word in response_text.split(" "):
                    yield {"data": word + " "}
                    await asyncio.sleep(0.01)

            return EventSourceResponse(greeting_generator())

    # ثانياً: البحث المباشر في ملفات JSON وبث النتيجة مباشرة
    async def json_generator():
        try:
            if not loaded_data:
                yield {
                    "data": (
                        "عذراً، لم يتم تحميل أي بيانات من ملفات JSON في"
                        " النظام."
                    )
                }
                return

            # إجراء البحث في خيط منفصل (Thread) لتجنب تجميد Event Loop
            matched_results = await asyncio.to_thread(
                search_in_json, user_query
            )

            if not matched_results:
                yield {
                    "data": (
                        "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في"
                        " البيانات المتاحة."
                    )
                }
                return

            # دمج النصوص المستخرجة
            extracted_text = "\n\n---\n\n".join(matched_results)

            # محاكاة البث (Streaming) كلمة بكلمة
            for word in extracted_text.split(" "):
                yield {"data": word + " "}
                await asyncio.sleep(0.005)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
