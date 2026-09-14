import os
import json
import random
import cohere
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cohere Intent API")

# 1. تهيئة عميل Cohere
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

patterns_list = []
intents_mapping = []
patterns_embeddings = None

# نموذج لاستقبال البيانات
class QueryRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_and_embed_intents():
    global patterns_list, intents_mapping, patterns_embeddings
    
    json_path = "intents.json"
    if not os.path.exists(json_path):
        print("Warning: intents.json file not found!")
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

@app.post("/predict")
def predict_intent(request: QueryRequest):
    user_text = request.text
    
    user_response = co.embed(
        texts=[user_text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    
    user_embedding = np.array(user_response.embeddings[0])
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    similarities = np.dot(patterns_embeddings, user_embedding)
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    THRESHOLD = 0.40
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    return {"response": selected_response, "score": float(best_score)}
