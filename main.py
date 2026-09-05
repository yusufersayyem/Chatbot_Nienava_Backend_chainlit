import json
import chainlit as cl
from rapidfuzz import fuzz, process

# ================= =================
# 1. تحميل قاعدة بيانات الأسئلة والأجوبة
# ===================================
DATA_FILE = "questions_answers.json"

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    # استخراج قائمة الأسئلة فقط لتسريع عملية المقارنة والبحث
    questions_list = [item["question"] for item in qa_data]
    print(f"✅ تم تحميل {len(qa_data)} سؤال وجواب بنجاح.")
except Exception as e:
    print(f"❌ خطأ في تحميل ملف JSON: {e}")
    qa_data = []
    questions_list = []


# ===================================
# 2. أحداث الباكند الخاصة بـ Chainlit
# ===================================


@cl.on_chat_start
async def start():
    """تنفذ هذه الدالة فور فتح المستخدم لواجهة الشات"""
    await cl.Message(
        content="أهلاً بك! أنا البوت التفاعلي لتربية نينوى. تفضل بطرح سؤالك وسأجيبك فوراً."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """تنفذ هذه الدالة في كل مرة يرسل فيها المستخدم رسالة"""
    user_query = message.content.strip()

    # التحقق من وجود بيانات
    if not qa_data:
        await cl.Message(
            content="عذراً، قاعدة البيانات غير متوفرة حالياً."
        ).send()
        return

    # منطق البحث والتحليل (Matching Engine)
    # يبحث عن أفضل تطابق بين سؤال المستخدم وقائمة الأسئلة
    match, score, index = process.extractOne(
        user_query, questions_list, scorer=fuzz.token_set_ratio
    )

    # تحديد نسبة القبول (مثلاً 55% أو أعلى)
    if score >= 55:
        # جلب الجواب الخاص بالسؤال المطابق
        bot_response = qa_data[index]["answer"]
    else:
        # رسالة تعذر العثور على إجابة
        bot_response = "عذراً، لم أجد إجابة دقيقة لسؤالك في قاعدة البيانات. يرجى التأكد من صياغة السؤال."

    # إرسال النتيجة إلى واجهة المستخدم
    await cl.Message(content=bot_response).send()
