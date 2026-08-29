import os
import re
import json
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from openai import AsyncOpenAI

app = FastAPI(title="RAG Streaming Backend - Nineveh Edu")

# ==========================================
# 1. إعداد المتغيرات والمفاتيح
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") # تم التغيير لـ Together AI Key
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# تهيئة عميل Together AI عبر AsyncOpenAI (يدعم OpenAI Compatibility)
llm_client = AsyncOpenAI(
    base_url="https://api.together.ai/v1",
    api_key=TOGETHER_API_KEY,
)

# ==========================================
# 2. كلاس الـ Embeddings الخاص بـ HuggingFace
# ==========================================
class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def _process_response(self, response) -> List[float]:
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

# ==========================================
# 3. تهيئة الـ Embeddings وقاعدة البيانات FAISS
# ==========================================
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# ==========================================
# 4. خريطة التحيات المباشرة
# ==========================================
GREETINGS_MAP = {
    r"^(مرحبا|مرحباً|أهلا|أهلاً|اهلين|أهلين|السلام عليكم|مرحبتين|هلا|صباح الخير|مساء الخير)": 
        "أهلاً بك! أنا المساعد الذكي للمديرية العامة لتربية نينوى. كيف يمكنني مساعدتك اليوم؟",
        
    r"^(من انت|من أنت|عرف عن نفسك|ما هو عملك|ماذا تفعل)": 
        "أنا مساعد ذكي مخصص للإجابة عن استفسارات الموظفين والمراجعين الخاصة بالمديرية العامة لتربية نينوى.",
        
    r"^(شكرا|شكراً|يعطيك العافية|تسلم|تسلم ايدك|مشكور)": 
        "العفو! أنا في الخدمة دائماً. هل لديك أي استفسار إداري آخر؟"
}

class QueryRequest(BaseModel):
    query: str

# ==========================================
# 5. نقطة النهاية (Endpoint) للبث المباشر
# ==========================================
@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    # أولاً: الرد السريع المباشر للتحيات
    for pattern, response_text in GREETINGS_MAP.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            async def greeting_generator():
                for word in response_text.split(" "):
                    yield {"data": word + " "}
                    await asyncio.sleep(0.02)
            return EventSourceResponse(greeting_generator())

    # ثانياً: البحث في FAISS ثم استدعاء نموذج GLM-5.3-Flash من Together AI
    try:
        docs = await asyncio.to_thread(
            vector_store.similarity_search, user_query, k=2
        )
        
        context = "\n\n".join([d.page_content for d in docs]) if docs else "لا يوجد سياق متوفر."

        # تعليمات حازمة لمنع كتابة الأكواد بجميع أشكالها
        system_instruction = f"""أنت مساعد خدمة العملاء والمراجعين في المديرية العامة لتربية نينوى.

تعليمات صارمة:
1. أجب باللغة العربية الفصحى وبطريقة إدارية ودودة وبسيطة.
2. يُمنع منعاً باتاً كتابة أي أسطر أو كتل برمجية (مثل Python, HTML, JS) أو استخدام التنسيق البرمجي (```).
3. أجب عن سؤال المستخدم بالاعتماد حصراً على السياق المرفق أدناه، وإذا لم توجد الإجابة وضّح ذلك بلباقة.

السياق المتاح:
{context}"""

        async def llm_generator():
            try:
                stream_response = await llm_client.chat.completions.create(
                    model="zai-org/GLM-5.3-Flash",  # تم تغيير النموذج إلى GLM-5.3-Flash
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query}
                    ],
                    stream=True
                )

                async for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        yield {"data": token}

            except Exception as inner_e:
                yield {"data": f"\n[خطأ في الاتصال بالنموذج: {str(inner_e)}]"}

        return EventSourceResponse(llm_generator())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
