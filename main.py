import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

import requests
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ==========================================
# 1. الإعدادات والتهيئات
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")
JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]

# ضع مفتاح HuggingFace الخاص بك هنا أو عبر متغيرات البيئة في Render
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "ضع_مفتاح_HUGGINGFACE_هنا")

# نموذج التضمين على Hugging Face
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None


def get_embeddings_from_hf(texts: List[str]) -> np.ndarray:
  """إرسال النصوص إلى Hugging Face والحصول على المتجهات."""
  response = requests.post(
      API_URL,
      headers=HEADERS,
      json={"inputs": texts, "options": {"wait_for_model": True}},
  )

  if response.status_code != 200:
    raise Exception(f"HF API Error: {response.status_code} - {response.text}")

  embeddings = np.array(response.json())
  # تطبيق Normalization للمتجهات
  norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
  return embeddings / np.maximum(norms, 1e-12)


async def prepare_and_embed_data():
  """تحميل النصوص واستدعاء الـ API لتوليد المتجهات عند الإقلاع."""
  global loaded_chunks, chunk_embeddings

  raw_texts = []
  for file_name in JSON_FILES:
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.exists(file_path):
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          content = json.load(f)
          items = content if isinstance(content, list) else [content]
          for item in items:
            if isinstance(item, dict):
              formatted_str = "\n".join([f"{k}: {v}" for k, v in item.items()])
            else:
              formatted_str = str(item)
            raw_texts.append(formatted_str)
      except Exception as e:
        print(f"❌ خطأ أثناء قراءة {file_name}: {e}")

  loaded_chunks = raw_texts

  if loaded_chunks:
    print(f"⏳ جاري جلب المتجهات لـ ({len(loaded_chunks)}) نص عبر API...")
    try:
      # تقسيم النصوص لدفعة بحجم 32 لتفادي حدود الطلب
      batch_size = 32
      all_embeddings = []
      for i in range(0, len(loaded_chunks), batch_size):
        batch = loaded_chunks[i : i + batch_size]
        batch_emb = await asyncio.to_thread(get_embeddings_from_hf, batch)
        all_embeddings.append(batch_emb)

      chunk_embeddings = np.vstack(all_embeddings)
      print("✅ تم جلب المتجهات بنجاح بدون استهلاك ذاكرة السيرفر!")
    except Exception as e:
      print(f"❌ خطأ أثناء جلب المتجهات: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
  asyncio.create_task(prepare_and_embed_data())
  yield


app = FastAPI(title="HuggingFace API Search - Nineveh Edu", lifespan=lifespan)

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
  if chunk_embeddings is None or len(loaded_chunks) == 0:
    return []

  # جلب متجه الاستعلام من API
  query_vec = get_embeddings_from_hf([query])[0]

  similarities = np.dot(chunk_embeddings, query_vec)
  top_indices = np.argsort(similarities)[::-1][:top_k]

  results = []
  for idx in top_indices:
    if similarities[idx] > 0.15:
      results.append(loaded_chunks[idx])

  return results


@app.post("/search-stream")
async def search_stream(req: QueryRequest):
  user_query = req.query.strip()

  async def json_generator():
    try:
      matched_results = await asyncio.to_thread(
          semantic_search, user_query, 3
      )

      if not matched_results:
        yield {
            "data": (
                "عذراً، لم أجد أي معلومات مطابقة لاستفسارك في البيانات"
                " المتاحة."
            )
        }
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
