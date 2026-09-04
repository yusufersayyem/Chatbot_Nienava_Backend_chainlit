import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# استيراد محرك ONNX والمحلل اللغوي فقط دون PyTorch
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

tokenizer = None
ort_session = None
loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None

def mean_pooling(model_output, attention_mask):
    """دالة حساب المتوسط الموزون للمتجهات عبر NumPy."""
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def encode_query_onnx(text: str) -> np.ndarray:
    """استخراج متجه النص باستخدام ONNX بحد أدنى للذاكرة."""
    # 1. التقطيع اللغوي (Tokenization)
    encoded_input = tokenizer(
        [text], 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="np"
    )
    
    # 2. التمرير في محرك ONNX الخفيف
    onnx_inputs = {
        "input_ids": encoded_input["input_ids"].astype(np.int64),
        "attention_mask": encoded_input["attention_mask"].astype(np.int64)
    }
    
    # تحسين التوافقية لو كانت هناك المدخلات الإضافية (token_type_ids)
    if "token_type_ids" in [inp.name for inp in ort_session.get_inputs()]:
        onnx_inputs["token_type_ids"] = encoded_input.get(
            "token_type_ids", 
            np.zeros_like(encoded_input["input_ids"])
        ).astype(np.int64)

    onnx_outputs = ort_session.run(None, onnx_inputs)
    
    # 3. حساب Pooled Embeddings و L2 Normalization
    sentence_embeddings = mean_pooling(onnx_outputs, encoded_input["attention_mask"])
    norm = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    return sentence_embeddings / np.maximum(norm, 1e-12)

def load_data():
    global tokenizer, ort_session, loaded_chunks, chunk_embeddings
    
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("❌ ملفات المتجهات غير موجودة! يرجى رفعها في المستودع.")

    print("⚡ جاري تحميل ملفات المتجهات المحسوبة...")
    chunk_embeddings = np.load(EMBEDDINGS_FILE)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)

    print("⏳ جاري تحميل المحلل اللغوي ومحرك ONNX الصغير جداً...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # استخدام ملف ONNX المكمّم صراحة عبر onnxruntime المباشر
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id=MODEL_NAME, 
        filename="onnx/model_quint8_avx2.onnx"
    )
    
    # إنشاء جلسة تشغيل خفيفة جداً بدون خيوط معالجة زائدة
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(model_path, opts)
    
    print("✅ تم تحميل الباك إند بنجاح باستهلاك ذاكرة ضئيل جداً (< 150MB)!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_data)
    yield

app = FastAPI(title="Ultra Low Memory Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def semantic_search(query: str, top_k: int = 3) -> List[str]:
    if ort_session is None or chunk_embeddings is None:
        return []

    query_vec = encode_query_onnx(query)[0]
    similarities = np.dot(chunk_embeddings, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.10:
            results.append(loaded_chunks[idx])

    return results

@app.post("/search-stream")
async def search_stream(req: QueryRequest):
    user_query = req.query.strip()

    async def json_generator():
        try:
            matched_results = await asyncio.to_thread(semantic_search, user_query, 3)

            if not matched_results:
                yield {"data": "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات المتاحة."}
                return

            extracted_text = "\n\n---\n\n".join(matched_results)
            words = extracted_text.split(" ")
            
            for i in range(0, len(words), 2):
                chunk = " ".join(words[i : i + 2]) + " "
                yield {"data": chunk}
                await asyncio.sleep(0.01)

        except Exception as err:
            yield {"data": f"\n⚠️ [حدث خطأ أثناء البحث: {str(err)}]"}

    return EventSourceResponse(json_generator())
