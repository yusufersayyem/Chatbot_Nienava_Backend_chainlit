import os
from typing import List, AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from huggingface_hub import AsyncInferenceClient, InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# 1. تهيئة تطبيق FastAPI وإعدادات CORS
app = FastAPI(title="RAG Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. المفاتيح والمعايير
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 3. فئة توليد الـ Embeddings
class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.feature_extraction(texts)
        return response.tolist() if hasattr(response, "tolist") else response

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text)
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            if isinstance(response[0][0], list):
                response = response[0][0]
            else:
                response = response[0]
        return response.tolist() if hasattr(response, "tolist") else response

# 4. تهيئة عملاء Hugging Face وتحميل قاعدة FAISS
llm_client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)

vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 5. نماذج الطلب (Schemas)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

# 6. دالة البث التدفقي والمعالجة (تم إصلاح خطأ الدمج هنا)
async def generate_chat_stream(message: str, history: List[ChatMessage]) -> AsyncGenerator[str, None]:
    try:
        # البحث الدلالي في FAISS
        docs = vector_store.similarity_search(message, k=2)
        
        # تحويل المحتوى إلى str صراحة لتفادي خطأ (can only concatenate str to str)
        context_text = "\n\n".join([str(doc.page_content) for doc in docs]) if docs else ""

        # بناء نص الإدخال للنموذج
        if context_text:
            user_prompt = f"المعلومات المستخرجة من قاعدة البيانات:\n{context_text}\n\nسؤال المستخدم: {message}"
        else:
            user_prompt = message

        messages_for_llm = [
            {
                "role": "system", 
                "content": "أنت مساعد ذكي ومفيد. اعتمِد على السياق المرفق للإجابة عن أسئلة المستخدم بوضوح ودقة. إذا لم تجد الإجابة في السياق، أجب بما تعرفه بشكل عام."
            }
        ]
        
        for msg in history:
            messages_for_llm.append({"role": msg.role, "content": msg.content})
            
        messages_for_llm.append({"role": "user", "content": user_prompt})

        # الاتصال بنموذج Qwen بالبث التدفقي
        stream = await llm_client.chat_completion(
            messages=messages_for_llm,
            max_tokens=2048,
            temperature=0.3,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"\n[حدث خطأ أثناء معالجة الطلب: {str(e)}]"

# 7. المسارات (Endpoints)
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return EventSourceResponse(
        generate_chat_stream(request.message, request.history),
        media_type="text/event-stream"
    )

@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running successfully!"}

# 8. التشغيل التلقائي المتوافق مع Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
