import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_mistralai import ChatMistralAI

load_dotenv()

app = FastAPI(title="RAG API - Nineveh & Mosul Uni")

# 1. إعداد المتغيرات البيئية
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
COLLECTION_NAME = "my_pdf_documents"

def format_docs(docs):
    """
    قراءة الـ metadata الخاصة بكل مستند واستخراج السؤال والجواب بشكل منظم
    لضمان توفير أفضل سياق وضوحاً للـ LLM
    """
    formatted_chunks = []
    for doc in docs:
        meta = doc.metadata
        # استخراج البيانات من الـ metadata مع الاعتماد على page_content كخيار احتياطي
        question = meta.get("question")
        answer = meta.get("answer")
        
        if question and answer:
            formatted_chunks.append(f"السؤال المرجعي: {question}\nالجواب المرجعي: {answer}")
        else:
            formatted_chunks.append(doc.page_content)
            
    return "\n\n---\n\n".join(formatted_chunks)

def load_rag_chain():
    # 2. إعداد نموذج التضمين عبر Hugging Face Endpoint
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=HF_TOKEN
    )
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    
    # 3. إعداد المسترجع (Retriever)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'k': 3,
            'score_threshold': 0.55
        }
    )
    
    # 4. إعداد نموذج اللغة (Mistral AI)
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.0, 
        max_retries=2,
        streaming=True
    )
    
    # 5. تعليمات النظام (System Prompt)
    system_prompt = (
        "أنت مساعد رقمي رسمي للإجابة عن استفسارات مديرية تربية نينوى وجامعة الموصل.\n"
        "التزم بالقواعد التالية بدقة متناهية:\n\n"
        "1. **التحيات المجرّدة (بدون سؤال):**\n"
        "   - إذا كانت الرسالة تحية فقط (مثل: 'السلام عليكم' أو 'مرحباً')، أجب بـ: 'وعليكم السلام ورحمة الله وبركاته. أهلاً بك، كيف يمكنني مساعدتك اليوم؟'\n\n"
        "2. **الأسئلة والاستفسارات (سواء اقترنت بتحية أم لا):**\n"
        "   - **يُحظر تماماً** البدء بعبارة 'السلام عليكم ورحمة الله وبركاته' عند الإجابة على أي سؤال أو استفسار.\n"
        "   - ادخل في الإجابة المباشرة عن السؤال فوراً وبأسلوب رسمي اعتماداً على السياق المتاح فقط.\n\n"
        "3. **الشكر والثناء:**\n"
        "   - إذا كانت الرسالة عبارة شكر، أجب بـ: 'العفو، هذا واجبي وفي الخدمة دائماً.'\n\n"
        "4. **قيود السياق:**\n"
        "   - اعتمد **فقط وحصراً** على أزواج (الأسئلة والأجوبة) المرفقة في السياق.\n"
        "   - إذا لم تجد إجابة ضمن السياق، أجب بـ: 'عذراً، لا تتوفر معلومة رسمية خاصة بهذا الاستفسار ضمن التعليمات المتاحة حالياً.'\n\n"
        "السياق المتاح (أسئلة وأجوبة رسمية):\n{context}\n\n"
        "سؤال/رسالة المستخدم: {question}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # 6. بناء السلسلة باستخدام LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

rag_chain = load_rag_chain()

class QueryRequest(BaseModel):
    question: str

@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    async def generate():
        async for chunk in rag_chain.astream(request.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
