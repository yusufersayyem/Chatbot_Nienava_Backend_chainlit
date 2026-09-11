import os
import json
import random
import cohere
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Cohere Intent Classifier API")

# 1. تهيئة عميل Cohere
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# متغيرات التخزين
patterns_list = []
intents_mapping = []
patterns_embeddings = None

def load_and_embed_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, patterns_embeddings
    
    if not os.path.exists(json_path):
        print(f"تنبيه: لم يتم العثور على الملف {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            patterns_list.append(pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    print("جاري استدعاء Cohere API لحساب متجهات الـ patterns...")
    
    response = co.embed(
        texts=patterns_list,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    
    embeddings_matrix = np.array(response.embeddings)
    patterns_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    print("تم تجهيز متجهات الـ patterns بنجاح!")

# تحميل البيانات عند تشغيل الخادم
@app.on_event("startup")
async def startup_event():
    load_and_embed_intents()

# نموذج طلب البيانات للمستخدم
class QueryRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_intent(request: QueryRequest):
    if patterns_embeddings is None:
        raise HTTPException(status_code=500, detail="نموذج المتجهات غير جاهز بعد.")

    user_text = request.text
    
    # تحويل نص المستخدم إلى متجه
    user_response = co.embed(
        texts=[user_text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    
    user_embedding = np.array(user_response.embeddings[0])
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    # حساب التشابه الدلالي
    similarities = np.dot(patterns_embeddings, user_embedding)
    best_match_idx = np.argmax(similarities)
    best_score = float(similarities[best_match_idx])
    
    THRESHOLD = 0.40
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
        tag = matched_intent["tag"]
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        tag = "unknown"
        
    return {
        "response": selected_response,
        "score": best_score,
        "tag": tag
    }
