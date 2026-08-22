import os
import asyncio
from typing import List
import chainlit as cl
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# 1. إعداد المتغيرات والمفاتيح
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 2. كلاس الـ Embeddings
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

# 3. تهيئة النموذج وقاعدة البيانات
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)

vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

@cl.on_message
async def main(message: cl.Message):
    try:
        # البحث عن أحدث وأقرب إجابة لسؤال المستخدم (k=1 للوصول لأدق إجابة مباشرة)
        # يمكنك زيادة k إلى 2 أو 3 إذا أردت عرض أكثر من خيار للمستخدم
        docs = await asyncio.to_thread(
            vector_store.similarity_search, message.content, k=1
        )

        if docs:
            # عرض الإجابة المستخرجة مباشرة
            answer = docs[0].page_content
            await cl.Message(content=answer).send()
        else:
            await cl.Message(
                content="عذراً، هذه المعلومة غير متوفرة في قاعدة البيانات المتاحة لدي."
            ).send()

    except Exception as e:
        await cl.Message(content=f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}").send()
