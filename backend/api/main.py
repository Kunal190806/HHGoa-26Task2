from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import time

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.retrieval import RetrievalPipeline
from pipeline.embeddings import EmbeddingPipeline
from pipeline.vector_store import VectorStore
from pipeline.query_router import QueryRouter
from pipeline.generation import GenerationPipeline
from pipeline.grounding import GroundingValidator
from .sarvam_client import SarvamClient

app = FastAPI(title="HH Goa 2026 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Pipelines (This may take a moment to load models)...")
# Initialize singletons to keep models in memory
try:
    embedder = EmbeddingPipeline()
    store = VectorStore(collection_name="msmarco_xi", dense_dim=embedder.dense_dim)
    retriever = RetrievalPipeline(embedder, store)
    router = QueryRouter()
    generator = GenerationPipeline(provider_name="gemini") # Or dynamic based on env
    grounder = GroundingValidator()
    stt_client = SarvamClient()
    print("Pipelines initialized successfully.")
except Exception as e:
    print(f"Error initializing pipelines: {e}")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "HH Goa 2026 API is running"}

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    return process_rag_pipeline(request.query)

@app.post("/api/voice_ask")
async def ask_voice_question(audio: UploadFile = File(...)):
    # 1. Speech-to-Text
    t0 = time.perf_counter()
    audio_bytes = await audio.read()
    transcript = stt_client.transcribe_audio(audio_bytes, audio.filename)
    stt_latency = (time.perf_counter() - t0) * 1000
    
    if "error" in transcript.lower() or not transcript.strip():
        return {
            "status": "error",
            "message": "We couldn't transcribe that audio. Please try again.",
            "latency": {"stt_ms": stt_latency}
        }
        
    # Process through standard RAG pipeline
    response = process_rag_pipeline(transcript)
    response["latency_metrics"]["stt_ms"] = stt_latency
    response["latency_metrics"]["total_e2e_ms"] = response["latency_metrics"]["total_e2e_ms"] + stt_latency
    return response

def process_rag_pipeline(query: str):
    t_start = time.perf_counter()
    metrics = {}
    
    # 1. Query Router
    t0 = time.perf_counter()
    routing_info = router.route_query(query)
    metrics["query_routing_ms"] = (time.perf_counter() - t0) * 1000
    
    # 1.5 Chitchat guard: respond directly without retrieval
    if routing_info["intent"] == "Chitchat":
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        return {
            "status": "success",
            "query": query,
            "answer": "Hello! I'm your multilingual RAG assistant. Ask me a question about the dataset and I'll find the answer for you! 🙏",
            "sources": [],
            "routing": routing_info,
            "retrieval_confidence": "N/A",
            "grounding": {"status": "SKIP", "reason": "Chitchat query — no retrieval needed."},
            "latency_metrics": metrics
        }
    
    # 2. Hybrid Retrieval & Reranking
    retrieval_res = retriever.retrieve(query, routing_info["strategy"])
    retrieval_metrics = retrieval_res["latency_ms"]
    
    # 3. Context Sufficiency Check
    confidence = retrieval_res["confidence"]
    if confidence == "LOW":
        # Guardrail: Refuse to answer if confidence is too low
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        return {
            "status": "refusal",
            "query": query,
            "routing": routing_info,
            "confidence": confidence,
            "message": "I couldn't find enough reliable information in the dataset to answer that reliably.",
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        
    # 4. LLM Generation
    t0 = time.perf_counter()
    answer = generator.generate_answer(query, retrieval_res["results"])
    metrics["generation_ms"] = (time.perf_counter() - t0) * 1000
    
    # 5. Lightweight Grounding
    t0 = time.perf_counter()
    grounding_status, grounding_reason = grounder.validate(answer, retrieval_res["results"], llm_client=generator.provider)
    metrics["grounding_ms"] = (time.perf_counter() - t0) * 1000
    
    if grounding_status == "FAIL":
        metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
        return {
            "status": "refusal",
            "query": query,
            "routing": routing_info,
            "confidence": confidence,
            "message": "I generated an answer but failed the grounding check, so I cannot provide it.",
            "latency_metrics": {**metrics, **retrieval_metrics}
        }
        
    metrics["total_e2e_ms"] = (time.perf_counter() - t_start) * 1000
    
    # Format sources
    sources = []
    for r in retrieval_res["results"]:
        sources.append({
            "relevance": round(r.get("rerank_score", 0), 2),
            "language": r["payload"].get("language", "en"),
            "strategy": r["payload"].get("chunk_strategy", "unknown"),
            "document_id": r["payload"].get("document_id", ""),
            "text": r["payload"]["text"]
        })
        
    return {
        "status": "success",
        "query": query,
        "answer": answer,
        "sources": sources,
        "routing": routing_info,
        "retrieval_confidence": confidence,
        "grounding": {
            "status": grounding_status,
            "reason": grounding_reason
        },
        "latency_metrics": {**metrics, **retrieval_metrics}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
