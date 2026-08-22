import os
import re
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Streaming Backend")

# 1. إعداد المتغيرات والمفاتيح
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 2. كلاس الـ Embeddings الخاص بـ HuggingFace
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

# 3. تهيئة النموذج وقاعدة البيانات عند بدء التشغيل
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 4. خريطة التعبير النمطي (Regex) للتعرف على التحيات والأسئلة الشائعة
GREETINGS_MAP = {
    r"^(مرحبا|مرحباً|أهلا|أهلاً|اهلين|أهلين|السلام عليكم|مرحبتين|هلا|صباح الخير|مساء الخير)": 
        "أهلاً بك! كيف يمكنني مساعدتك اليوم في البحث داخل قاعدة البيانات؟",
        
    r"^(من انت|من أنت|عرف عن نفسك|ما هو عملك|ماذا تفعل)": 
        "أنا مساعد ذكي مخصص للبحث في المستندات والرد على استفساراتك بناءً على البيانات المتاحة.",
        
    r"^(شكرا|شكراً|يعطيك العافية|تسلم|تسلم ايدك|مشكور)": 
        "العفو! أنا في الخدمة دائماً. هل لديك أي استفسار آخر؟"
}

class QueryRequest(BaseModel):
    query: str

@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    # أولاً: الفحص والرد الفوري إذا كان المدخل تحية أو سؤال عام
    for pattern, response_text in GREETINGS_MAP.items():
        if re.search(pattern, user_query, re.IGNORECASE):
            async def greeting_generator():
                for word in response_text.split(" "):
                    yield {"data": word + " "}
                    await asyncio.sleep(0.03)  # سرعة تدفّق الكلمات
            return EventSourceResponse(greeting_generator())

    # ثانياً: إذا لم تكن تحية، الانتقال للبحث الفعلي في FAISS
    try:
        docs = await asyncio.to_thread(
            vector_store.similarity_search, user_query, k=1
        )
        
        async def event_generator():
            if docs:
                full_text = docs[0].page_content
                words = full_text.split(" ")
                for word in words:
                    yield {"data": word + " "}
                    await asyncio.sleep(0.03)
            else:
                yield {"data": "عذراً، هذه المعلومة غير متوفرة في قاعدة البيانات المتاحة لدي."}

        return EventSourceResponse(event_generator())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
