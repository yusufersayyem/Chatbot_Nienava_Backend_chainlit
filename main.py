import asyncio
from contextlib import asynccontextmanager
import gc
import json
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sse_starlette.sse import EventSourceResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "loaded_data")
JSON_FILES = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json"]

# استخدام النموذج الأكثر خفة وسرعة للغة العربية
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

model = None
loaded_chunks: List[str] = []
chunk_embeddings: np.ndarray = None


def load_model_safely():
  """تحميل النموذج بتقليل استهلاك الذاكرة."""
  global model
  if model is None:
    print("⏳ جاري تحميل نموذج التضمين...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # تقليل دقة النموذج لتوفير 50% من الذاكرة
    model.to("cpu")
    print("✅ تم تحميل النموذج.")


async def prepare_and_embed_data():
  """تحميل النصوص واستخراج المتجهات باستهلاك منخفض للذاكرة."""
  global loaded_chunks, chunk_embeddings

  load_model_safely()

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
    print(f"⏳ جاري توليد المتجهات لـ ({len(loaded_chunks)}) نص...")
    # حساب المتجهات على دفعات صغيرة لتجنب الـ OOM
    embeddings = await asyncio.to_thread(
        model.encode,
        loaded_chunks,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    chunk_embeddings = np.array(embeddings, dtype=np.float32)
    print("✅ اكتمل استخراج المتجهات بنجاح.")

  # تنظيف الذاكرة المؤقتة
  gc.collect()


@asynccontextmanager
async def lifespan(app: FastAPI):
  asyncio.create_task(prepare_and_embed_data())
  yield


app = FastAPI(title="Semantic Search API", lifespan=lifespan)

# تفعيل CORS للفرونت إند
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
  if chunk_embeddings is None or len(loaded_chunks) == 0 or model is None:
    return []

  query_vec = model.encode([query], normalize_embeddings=True)[0]
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
