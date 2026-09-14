import os
import json
import random
import cohere
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache

app = FastAPI(title="Cohere Intent API Optimized")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

intents_mapping = []
patterns_embeddings = None

class QueryRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_data():
    global intents_mapping, patterns_embeddings
    
    # 1. تحميل خرائط الـ Intents
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    # 2. تحميل المتجهات مسبقة الحساب لتوفير وقت الإقلاع وطلبات الـ API
    if os.path.exists("patterns_embeddings.npy"):
        patterns_embeddings = np.load("patterns_embeddings.npy")
    else:
        print("Warning: patterns_embeddings.npy not found! Please pre-compute it.")

# Caching لتوفير تكلفة استدعاء الـ API للأسئلة المكررة
@lru_cache(maxsize=2048)
def get_user_embedding(text: str):
    response = co.embed(
        texts=[text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    user_emb = np.array(response.embeddings[0])
    return user_emb / np.linalg.norm(user_emb)

@app.post("/predict")
def predict_intent(request: QueryRequest):
    user_text = request.text.strip().lower()
    
    if not user_text or patterns_embeddings is None:
        return {"response": "حدث خطأ في النظام، يرجى المحاولة لاحقاً.", "score": 0.0}
    
    # جلب المتجه (سواء من الكاش أو من الـ API)
    user_embedding = get_user_embedding(user_text)
    
    # حساب التشابه
    similarities = np.dot(patterns_embeddings, user_embedding)
    best_match_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_match_idx])
    
    THRESHOLD = 0.40
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    return {"response": selected_response, "score": best_score}
