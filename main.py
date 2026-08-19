import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. تحميل البيئة
load_dotenv()

app = FastAPI(title="Nineveh Education RAG API")

# إتاحة CORS لاتصال الفرونت إند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "my_pdf_documents"

RAG_RESOURCES = None

# دالة لتنسيق النصوص المسترجعة من Qdrant ليمررها للموديل
def format_docs(docs):
    formatted_chunks = []
    for doc in docs:
        meta = doc.metadata
        question = meta.get("question", doc.page_content)
        answer = meta.get("answer", "")
        if answer:
            formatted_chunks.append(f"السؤال المرجعي: {question}\nالجواب المرجعي: {answer}")
        else:
            formatted_chunks.append(doc.page_content)
    return "\n\n---\n\n".join(formatted_chunks)

# دالة التهيئة المتأخرة (Lazy Initialization)
def init_resources():
    global RAG_RESOURCES
    if RAG_RESOURCES is not None:
        return RAG_RESOURCES

    # نفس نموذج التضمين المستخدم بالرفعة لتطابق 100%
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )
    
    # مسترجع البيانات
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    
    # Groq المجاني والسريع
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=GROQ_API_KEY,
        streaming=True
    )
    
    system_prompt = (
        "أنت مساعد رقمي رسمي للإجابة عن استفسارات المديرية العامة لتربية نينوى.\n"
        "التزم بالقواعد التالية بدقة متناهية:\n\n"
        "1. ادخل في الإجابة المباشرة عن السؤال فوراً وبأسلوب رسمي اعتماداً على السياق المتاح فقط.\n"
        "2. اعتمد فقط وحصراً على أزواج (الأسئلة والأجوبة) المرفقة في السياق.\n"
        "3. إذا لم تجد إجابة صريحة ضمن السياق المرفق، أجب بـ: "
        "'عذراً، لا تتوفر معلومة رسمية خاصة بهذا الاستفسار ضمن التعليمات المتاحة حالياً.'\n\n"
        "السياق المتاح:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    
    RAG_RESOURCES = {
        "retriever": retriever,
        "prompt": prompt,
        "llm": llm
    }
    return RAG_RESOURCES

# نماذج طلبات API
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.get("/")
def read_root():
    return {"status": "FastAPI Server is running!"}

# كاش فحص التحيات والشكر السريعة (توفيراً للحدود المجانية)
def check_direct_intents(user_input: str) -> Optional[str]:
    text = user_input.strip().lower()
    greetings = ["السلام عليكم", "مرحبا", "مرحباً", "اهلا", "أهلاً", "صباح الخير", "مساء الخير"]
    thanks = ["شكرا", "شكراً", "مشكور", "رحم الله والديك", "تسلم"]
    
    if text in greetings:
        return "وعليكم السلام ورحمة الله وبركاته. أهلاً بك، كيف يمكنني مساعدتك في تعليمات تربية نينوى اليوم؟"
    if text in thanks:
        return "العفو، أنا في الخدمة دائماً لأي استفسار رسمي."
    return None

# مسار المحادثة الرئيسي مع Streaming
@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    # 1. رد مباشر إذا كانت تحية أو شكر
    direct_response = check_direct_intents(request.question)
    if direct_response:
        async def generate_direct():
            yield direct_response
        return StreamingResponse(generate_direct(), media_type="text/event-stream")

    # 2. تشغيل الـ RAG
    resources = init_resources()
    retriever = resources["retriever"]
    prompt = resources["prompt"]
    llm = resources["llm"]

    # جلب المستندات بناءً على السؤال
    docs = await retriever.ainvoke(request.question)
    context_text = format_docs(docs)

    # تجهيز ذاكرة المحادثة السابقة
    formatted_history = [
        (msg.role if msg.role != "user" else "human", msg.content) 
        for msg in request.history
    ]

    chain = prompt | llm | StrOutputParser()
    
    # البث المباشر للإجابة حرفاً بحرف
    async def generate():
        async for chunk in chain.astream({
            "context": context_text,
            "question": request.question,
            "history": formatted_history
        }):
            yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")
