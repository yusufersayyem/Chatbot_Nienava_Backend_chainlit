import asyncio
import os
import re
from typing import List

from fastapi import FastAPI
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="FAISS Search Backend - Nineveh Edu")

# ==========================================
# 1. إعداد المتغيرات والمفاتيح
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"


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
# 5. نقاط النهاية (Endpoints)
# ==========================================


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is active"}


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

    # ثانياً: البحث المباشر في FAISS وبث النتيجة مباشرة
    async def faiss_generator():
        try:
            if not vector_store:
                yield {
                    "data": (
                        "عذراً، قاعدة البيانات (FAISS Index) غير متوفرة"
                        " حالياً."
                    )
                }
                return

            # إتمام البحث في خيط منفصل (Thread) لتجنب تجميد Event Loop
            docs = await asyncio.to_thread(
                vector_store.similarity_search, user_query, k=3
            )

            if not docs:
                yield {
                    "data": (
                        "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في"
                        " المستندات المتاحة."
                    )
                }
                return

            # دمج النصوص المستخرجة من FAISS
            extracted_text = "\n\n---\n\n".join([d.page_content for d in docs])

            # محاكاة البث (Streaming) كلمة بكلمة لتبقى واجهة الـ Frontend تعمل بسلاسة
            for word in extracted_text.split(" "):
                yield {"data": word + " "}
                await asyncio.sleep(0.005)  # سرعة البث

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(faiss_generator())
