import asyncio
import os
import re
from typing import List
from fastapi import FastAPI
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from openai import AsyncOpenAI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="RAG Streaming Backend - Nineveh Edu")

# ==========================================
# 1. إعداد المتغيرات والمفاتيح
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # مفتاح OpenRouter

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 🟢 التعديل 1: تهيئة عميل AsyncOpenAI ليتصل بـ OpenRouter بدلاً من Hugging Face
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        # ترويسات اختيارية يفضل إضافتها وفق توثيق OpenRouter
        "HTTP-Referer": "https://nineveh-edu.gov.iq",
        "X-Title": "Nineveh Edu Chatbot",
    },
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

        while (
            isinstance(response, list)
            and len(response) > 0
            and isinstance(response[0], list)
        ):
            if isinstance(response[0][0], list):
                response = response[0]
            else:
                break

        if (
            isinstance(response, list)
            and len(response) > 0
            and isinstance(response[0], list)
        ):
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
try:
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"تحذير: تعذر تحميل FAISS Index: {e}")
    vector_store = None

# ==========================================
# 4. خريطة التحيات المباشرة
# ==========================================
GREETINGS_MAP = {
    r"^(مرحبا|مرحباً|أاهلا|أهلاً|اهلين|أهلين|السلام عليكم|مرحبتين|هلا|صباح الخير|مساء الخير)": (
        "أهلاً بك! أنا المساعد الذكي للمديرية العامة لتربية نينوى. كيف"
        " يمكنني مساعدتك اليوم؟"
    ),
    r"^(من انت|من أنت|عرف عن نفسك|ما هو عملك|ماذا تفعل)": (
        "أنا مساعد ذكي مخصص للإجابة عن استفسارات الموظفين والمراجعين الخاصة"
        " بالمديرية العامة لتربية نينوى."
    ),
    r"^(شكرا|شكراً|يعطيك العافية|تسلم|تسلم ايدك|مشكور)": (
        "العفو! أنا في الخدمة دائماً. هل لديك أي استفسار إداري آخر؟"
    ),
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

    # ثانياً: البحث في FAISS ثم استدعاء النموذج عبر مولد الـ SSE
    async def llm_generator():
        try:
            # البحث في FAISS
            context = "لا يوجد سياق متوفر."
            if vector_store:
                try:
                    docs = await asyncio.to_thread(
                        vector_store.similarity_search, user_query, k=3
                    )
                    if docs:
                        context = "\n\n".join([d.page_content for d in docs])
                except Exception as faiss_err:
                    context = f"[خطأ أثناء البحث في البيانات: {str(faiss_err)}]"

            system_instruction = f"""أنت مساعد رسمي مخصص لخدمة العملاء والمراجعين في المديرية العامة لتربية نينوى.

تعليمات صارمة يجب الالتزام بها:
1. أجب باللغة العربية الفصحى وبأسلوب إداري ورسمي ومؤدب.
2. اعتمد حصراً على "السياق المتاح" أدناه للإجابة عن السؤال. إذا لم توجد الإجابة في السياق، قل بلباقة: "عذراً، هذه المعلومة غير متوفرة لدي حالياً في التعليمات المتاحة."
3. يُمنع منعاً باتاً اختلاق أي معلومات من خارج السياق.
4. يُمنع منعاً باتاً كتابة أي أكواد برمجية (مثل Python, HTML) أو إشارات تقنية.

السياق المتاح:
{context}"""

            # 🟢 التعديل 2: استدعاء نموذج Ling 3.0 عبر OpenRouter
            response_stream = await openrouter_client.chat.completions.create(
                model="inclusionai/ling-3.0-flash-fin:free",  # اسم الموديل على OpenRouter
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.2,
                max_tokens=2048,
                stream=True,
                # 🟢 التعديل 3: إضافة خاصية الـ reasoning إن كنت تريد تفعيل قدرات التفكير الفائق في الموديل
                extra_body={"reasoning": {"enabled": True}},
            )

            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield {"data": content}

        except Exception as inner_e:
                    yield {"data": f"\n⚠️ [حدث خطأ في معالجة الطلب: {str(inner_e)}]"}

    return EventSourceResponse(llm_generator())
