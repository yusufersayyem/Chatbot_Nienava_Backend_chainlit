import os
import asyncio
from typing import List
import chainlit as cl
from huggingface_hub import AsyncInferenceClient, InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# 1. إعداد المتغيرات والمفاتيح
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 2. إنشاء تعليمات النظام الصارمة (System Prompt)
SYSTEM_PROMPT = """أنت مساعد آلي متخصص ومباشر. مهمتك الوحيدة هي الإجابة عن أسئلة المستخدم بناءً على "المعلومات المستخرجة من قاعدة البيانات" المرفقة في الرسالة فقط.

اتبع القواعد الصارمة التالية أثناء الإجابة:
1. يمنع منعاً باتاً تقديم أي معلومات من معرفتك العامة أو التخمين خارج النص المرفق.
2. إذا كانت الإجابة غير موجودة بوضوح ودقة في النص المرفق، أجب بحرفية: "عذراً، هذه المعلومة غير متوفرة في قاعدة البيانات المتاحة لدي."
3. انتبه بدقة متناهية للأرقام ونوع العقود (مثل: عقد 9000 وعقد 2500). لا تخلط إطلاقاً بين الشروط أو الإجازات الخاصة بكل عقد، وافصل بينهما بناءً على المذكور في النص فقط.
4. اذكر المعلومة المتعلقة بنوع العقد المطلوب في سؤال المستخدم حصراً."""

# 3. تحسين كلاس الـ Embeddings لتفادي أخطاء الأبعاد وانسداد الـ Event Loop
class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def _process_response(self, response) -> List[float]:
        """معالجة مخرجات feature_extraction وتسطيحها بالشكل الصحيح"""
        if hasattr(response, "tolist"):
            response = response.tolist()
        
        while isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            if isinstance(response[0][0], list):
                response = response[0]
            else:
                break

        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            response = [sum(col) / len(response) for col in zip(*response)]
            
        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text)
        return self._process_response(response)

# 4. تهيئة العملاء والقواعد
llm_client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)

# 5. تحميل قاعدة بيانات FAISS محلياً
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": SYSTEM_PROMPT}]
    )

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    try:
        # رفع عدد النتائج إلى k=4 لضمان استرجاع كافة السياقات ذات الصلة بالعقدين معاً إذا وجدا
        # استخدام asyncio.to_thread لمنع تجميد خادم التطبيق أثناء البحث
        docs = await asyncio.to_thread(
            vector_store.similarity_search, message.content, k=4
        )
        
        context_text = ""
        if docs:
            context_text = "\n---\n".join([doc.page_content for doc in docs])

        # صياغة المطالبة المباشرة للنموذج
        if context_text:
            user_prompt = (
                f"المعلومات المستخرجة من قاعدة البيانات:\n{context_text}\n\n"
                f"تنبيه: اقرأ المعلومات أعلاه بدقة وأجب عن السؤال المباشر دون الخلط بين أنواع العقود أو الإجازات.\n"
                f"سؤال المستخدم: {message.content}"
            )
        else:
            user_prompt = (
                f"سؤال المستخدم: {message.content}\n\n"
                f"ملاحظة: لا يوجد أي سياق مستخرج من قاعدة البيانات لهذا السؤال."
            )

        current_messages = message_history + [{"role": "user", "content": user_prompt}]

        msg = cl.Message(content="")
        await msg.send()

        # إرسال المحادثة مع تقليل temperature إلى 0.0 لإلغاء أي إبداع أو تخمين خارج السياق
        stream = await llm_client.chat_completion(
            messages=current_messages,
            max_tokens=2048,
            temperature=0.0,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()
        
        # حفظ السؤال الأصلي واستجابة المساعد للذاكرة بدون تخزين السياق الضخم
        message_history.append({"role": "user", "content": message.content})
        message_history.append({"role": "assistant", "content": msg.content})
        
        # تقليم السجل للحفاظ على حجم السياق (System Prompt + أحدث 10 رسائل)
        if len(message_history) > 11:
            message_history = [message_history[0]] + message_history[-10:]

        cl.user_session.set("message_history", message_history)

    except Exception as e:
        msg = cl.Message(content=f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}")
        await msg.send()
