import json
from rapidfuzz import fuzz, process

# 1. تحميل ملف الاسئلة والأجوبة
with open("questions_answers.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# استخراج الأسئلة فقط
questions_list = [item["question"] for item in qa_data]


def get_answer(user_query, threshold=55):
    """دالة البحث عن الإجابة الأكثر ملاءمة"""
    if not user_query.strip():
        return "يرجى كتابة سؤال!"

    # عملية البحث واختيار أفضل النتايج
    match, score, index = process.extractOne(
        user_query, questions_list, scorer=fuzz.token_set_ratio
    )

    if score >= threshold:
        return {
            "question": match,
            "answer": qa_data[index]["answer"],
            "score": round(score, 2),
        }
    else:
        return {
            "question": None,
            "answer": "عذراً، لم أجد إجابة دقيقة لسؤالك في قاعدة البيانات.",
            "score": round(score, 2),
        }


# 2. تجربة حية ومستمرة للمحادثة
print("🤖 أهلاً بك في بوت الاستفسارات! (اكتب 'خروج' للإغلاق)\n")

while True:
    user_input = input("👤 سؤالك: ")
    if user_input.strip().lower() in ["خروج", "exit", "quit"]:
        print("وداعاً!")
        break

    res = get_answer(user_input)

    if res.get("question"):
        print(f"🎯 السؤال المطابق: {res['question']}")
        print(f"💡 الجواب: {res['answer']}")
        print(f"📊 دقة التطابق: {res['score']}%\n")
    else:
        print(f"❌ {res['answer']}\n")
