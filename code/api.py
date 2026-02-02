import os
import json
import numpy as np
import faiss

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "code", "faiss_index")

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_NAME = "google/flan-t5-small"

app = FastAPI(title="Sanskrit RAG API")

class QueryRequest(BaseModel):
    query: str

# Load once at startup
index_path = os.path.join(INDEX_DIR, "index.faiss")
meta_path = os.path.join(INDEX_DIR, "metadata.json")

index = faiss.read_index(index_path)

with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embedder = SentenceTransformer(EMBED_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME)

def generate_answer(question: str, context: str):
    prompt = f"""
तुम् संस्कृत-सहायकः असि।
नियमाः:
1) केवलं दत्तसन्दर्भात् एव उत्तरं देहि।
2) यदि सन्दर्भे उत्तरं नास्ति, तर्हि एतदेव लिख: सन्दर्भे उत्तरं न लभ्यते।
3) उत्तरं संक्षेपेण (2-4 वाक्येषु) देहि।

सन्दर्भः:
{context}

प्रश्नः:
{question}

उत्तरम्:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

@app.get("/")
def root():
    return {"status": "ok", "message": "Sanskrit RAG API Running"}

@app.post("/ask")
def ask(req: QueryRequest):
    q = req.query.strip()

    q_vec = embedder.encode(q, normalize_embeddings=True).astype("float32")
    q_vec = np.expand_dims(q_vec, axis=0)

    k = 3
    scores, indices = index.search(q_vec, k)

    retrieved = []
    for idx in indices[0]:
        if idx == -1:
            continue
        retrieved.append(metadata[idx])

    context = "\n\n".join([r["text"] for r in retrieved])
    answer = generate_answer(q, context)

    return {
        "query": q,
        "answer": answer,
        "sources": [{"source": r["source"], "text": r["text"][:200]} for r in retrieved]
    }
