import os
import numpy as np
from crew_optimize import crew_optimize
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import time
from fastapi.responses import FileResponse
from phoenix_integration import trace_rag_session
from opentelemetry import trace
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBED_API = "http://localhost:11434"
OLLAMA_URL = os.getenv("OLLAMA_URL")
PG_VECTOR_DIM = int(os.getenv("PG_VECTOR_DIM", "384"))

tracer = trace.get_tracer("rag-server-app")
engine = create_engine(DATABASE_URL)
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
app = FastAPI(title="RAG Server for Open WebUI")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 6


def embed_text(text: str):
    if EMBED_API and "embeddings" in EMBED_API:
        r = requests.post(EMBED_API, json={"text": text})
        r.raise_for_status()
        return r.json()["embedding"]
    else:
        return bi_encoder.encode(text).tolist()


def retrieve_similar(embedding, k=6):
    query = text("""
        SELECT uuid,
               document,
               cmetadata,
               embedding <-> CAST(:vec AS vector) AS dist
        FROM langchain_pg_embedding
        ORDER BY embedding <-> CAST(:vec AS vector)
        LIMIT :k
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"vec": embedding, "k": k}).fetchall()
        candidates = []
        for r in rows:
            c = dict(r._mapping)
            # ensure cmetadata is parsed
            meta = c.get("cmetadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            c["cmetadata"] = meta
            candidates.append(c)
        return candidates


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict[str, Any]],
    bi_encoder: SentenceTransformer = None,
    cross_encoder: CrossEncoder = None
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    pairs = [(query, c["document"]) for c in candidates if c.get("document")]
    if not pairs:
        return candidates

    cross_scores = cross_encoder.predict(pairs)

    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    cand_embs = bi_encoder.encode([c["document"] for c in candidates], convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, cand_embs)[0]

    for c, cos, cross in zip(candidates, cos_scores, cross_scores):
        c["cosine_similarity"] = float(cos)
        c["cross_score"] = float(cross)
        # Debug prints
        print(f"Chunk: {c['document'][:80]}...")
        print(f"   Cosine Similarity: {c['cosine_similarity']:.4f}")
        print(f"   Cross Score:       {c['cross_score']:.4f}\n")

    candidates.sort(key=lambda x: x["cross_score"], reverse=True)
    return candidates


def build_prompt(query, context_chunks: List[Dict], system_instructions="You are a helpful assistant."):
    parts = []
    for c in context_chunks:
        meta = c.get("cmetadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except:
                meta = {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        text_chunk = c.get("document", "")
        parts.append(f"[{source} | page {page}]\n{text_chunk}")

    context_text = "\n\n---\n\n".join(parts)

    prompt = f"""{system_instructions}

Context:
{context_text}

User question:
{query}

Answer concisely and cite the sources in square brackets (e.g., [Procurement Manual.pdf page 21]).
If unsure, say 'I don't know' and propose how to find out.
"""
    return prompt


def call_ollama_generate(prompt, model="llama3:latest"):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def log_trace(session_id, query, context_chunks, response_text, meta):
    metadata = {**meta, "session_id": session_id}

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO rag_evaluation_logs (question, answer, contexts, metadata)
                VALUES (:question, :answer, :contexts, :metadata)
            """),
            {
                "question": query,
                "answer": response_text,
                "contexts": json.dumps([
                    {"document": c.get("document"), "cmetadata": c.get("cmetadata")}
                    for c in context_chunks
                ]),
                "metadata": json.dumps(metadata),
            }
        )


@app.get("/chatui")
def get_chatui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "webui_integration.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    query = req.message

    embedding = embed_text(query)
    retrieved = retrieve_similar(embedding, k=req.top_k)

    reranked = rerank_with_cross_encoder(
        query,
        retrieved,
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder
    )

    top_ctx = reranked[:4]

    prompt = build_prompt(query, top_ctx)
    optimized_prompt = crew_optimize(prompt)
    answer = call_ollama_generate(optimized_prompt, model=os.getenv("OLLAMA_MODEL", "llama3:latest"))

    log_trace(req.session_id, query, top_ctx, answer, {"elapsed": time.time() - start})

    elapsed = time.time() - start

    trace_rag_session(
        req.session_id,
        query,
        [c.get("document") for c in top_ctx],
        answer,
        elapsed,
        "llama3:latest"
    )

    response = {
        "answer": answer,
        "sources": [
            {
                "source": c.get("cmetadata", {}).get("source", "unknown") if isinstance(c.get("cmetadata"), dict) else "unknown",
                "page": c.get("cmetadata", {}).get("page", "?") if isinstance(c.get("cmetadata"), dict) else "?",
                "snippet": c.get("document", "")[:500]
            }
            for c in top_ctx
        ]
    }
    return response
