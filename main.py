import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
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

load_dotenv()

app = FastAPI(title="Nineveh Education RAG API")

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
executor = ThreadPoolExecutor(max_workers=1)

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

def _load_resources_sync():
    global RAG_RESOURCES
    if RAG_RESOURCES is not None:
        return RAG_RESOURCES

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
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
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

async def get_resources():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _load_resources_sync)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.get("/")
def read_root():
    return {"status": "FastAPI Server with Llama-3.3-70b is online!"}

def check_direct_intents(user_input: str) -> Optional[str]:
    text = user_input.strip().lower()
    greetings = ["السلام عليكم", "مرحبا", "مرحباً", "اهلا", "أهلاً", "صباح الخير", "مساء الخير"]
    thanks = ["شكرا", "شكراً", "مشكور", "رحم الله والديك", "تسلم"]
    
    if text in greetings:
        return "وعليكم السلام ورحمة الله وبركاته. أهلاً بك، كيف يمكنني مساعدتك في تعليمات تربية نينوى اليوم؟"
    if text in thanks:
        return "العفو، أنا في الخدمة دائماً لأي استفسار رسمي."
    return None

@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    direct_response = check_direct_intents(request.question)
    if direct_response:
        async def generate_direct():
            yield direct_response
        return StreamingResponse(generate_direct(), media_type="text/event-stream")

    try:
        resources = await get_resources()
        retriever = resources["retriever"]
        prompt = resources["prompt"]
        llm = resources["llm"]

        docs = await retriever.ainvoke(request.question)
        context_text = format_docs(docs)

        formatted_history = [
            (msg.role if msg.role != "user" else "human", msg.content) 
            for msg in request.history
        ]

        chain = prompt | llm | StrOutputParser()
        
        async def generate():
            async for chunk in chain.astream({
                "context": context_text,
                "question": request.question,
                "history": formatted_history
            }):
                yield chunk

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        async def generate_error():
            yield f"⚠️ حدث خطأ في معالجة الطلب داخل الباك إند: {str(e)}"
        return StreamingResponse(generate_error(), media_type="text/event-stream")
