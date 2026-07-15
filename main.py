import csv
import json
import math
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "documents.csv"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"
RERANKER_PATH = BASE_DIR / "reranker_scores.json"

with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    documents = list(reader)

with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
    embeddings = json.load(f)

with open(RERANKER_PATH, "r", encoding="utf-8") as f:
    reranker_scores = json.load(f)

for doc in documents:
    if "year" in doc:
        doc["year"] = int(doc["year"])

class VectorSearchRequest(BaseModel):
    query_id: str
    query_vector: List[float]
    top_k: int
    rerank_top_n: int
    filter: Dict[str, Any]

def matches_filter(doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for field, condition in filters.items():
        if field not in doc:
            return False

        value = doc[field]

        if isinstance(condition, dict):
            if "gte" in condition and value < condition["gte"]:
                return False
            if "lte" in condition and value > condition["lte"]:
                return False
            if "in" in condition and value not in condition["in"]:
                return False
        else:
            if value != condition:
                return False
    return True

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom

@app.get("/")
def home():
    return {"message": "Vector Search API is running"}

@app.post("/vector-search")
def vector_search(request: VectorSearchRequest):
    filtered_docs = [doc for doc in documents if matches_filter(doc, request.filter)]

    stage1 = []
    for doc in filtered_docs:
        doc_id = doc["doc_id"]
        if doc_id not in embeddings:
            continue
        sim = cosine_similarity(request.query_vector, embeddings[doc_id])
        stage1.append((doc_id, sim))

    stage1_sorted = sorted(stage1, key=lambda x: (-x[1], x[0]))
    top_k_docs = stage1_sorted[:request.top_k]

    query_scores = reranker_scores.get(request.query_id, {})
    stage2 = []
    for doc_id, _ in top_k_docs:
        rerank_score = query_scores.get(doc_id, 0.0)
        stage2.append((doc_id, rerank_score))

    stage2_sorted = sorted(stage2, key=lambda x: (-x[1], x[0]))
    final_docs = [doc_id for doc_id, _ in stage2_sorted[:request.rerank_top_n]]

    return {"matches": final_docs}