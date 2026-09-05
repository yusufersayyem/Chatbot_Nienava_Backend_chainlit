import asyncio
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rapidfuzz import fuzz, process

app = FastAPI(
    title="Nineveh Education Chatbot Backend",
    description="Backend service matching questions and streaming responses.",
)

# 1. السماح باتصالات CORS لتجنب مشاكل الاتصال مع الفرونت إند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تحميل ملف الأسئلة والأجوبة
DATA_FILE = "questions_answers.json"

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions_list = [item["question"] for item in qa_data]
    print(f"✅ تم تحميل {len(qa_data)} سؤال بنجاح.")
except Exception as e:
    print(f"❌ خطأ في تحميل ملف JSON: {e}")
    qa_data = []
    questions_list = []


# 3. تحديد كائن البيانات القادم من الفرونت إند
class QueryModel(BaseModel):
    query: str


# 4. دالة توليد البث المباشر (SSE Stream Generator)
async def generate_response_stream(user_query: str):
    if not qa_data:
        err_msg = json.dumps(
            {"data": "عذراً، قاعدة البيانات غير متوفرة حالياً."}
        )
        yield f"data: {err_msg}\n\n"
        yield "data: [DONE]\n\n"
        return

    # المطابقة باستخدام RapidFuzz
    match, score, index = process.extractOne(
        user_query, questions_list, scorer=fuzz.token_set_ratio
    )

    if score >= 55:
        response_text = qa_data[index]["answer"]
    else:
        response_text = (
            "عذراً، لم أجد إجابة دقيقة لسؤالك. يرجى التأكد من صياغة السؤال."
        )

    # محاكاة البث المباشر (إرسال النص كلمة بكلمة ليعطي مظهراً تفاعلياً ممتازاً)
    words = response_text.split(" ")
    for word in words:
        payload = json.dumps({"data": word + " "}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        await asyncio.sleep(0.04)  # تأخير زمني بسيط بين الكلمات

    # إشارة انتهاء البث المقروءة في الفرونت إند
    yield "data: [DONE]\n\n"


# 5. المسار المطابق تماماً لما يطلبه الفرونت إند
@app.post("/search-stream")
async def search_stream(payload: QueryModel):
    return StreamingResponse(
        generate_response_stream(payload.query), media_type="text/event-stream"
    )


# نقطة فحص الصحة للسيرفر على Render
@app.get("/")
def health_check():
    return {"status": "Backend is up and running!"}
