import time
from typing import Dict, Any, List, Tuple
from pipeline.vector_store import VectorStore
from pipeline.embeddings import EmbeddingPipeline
from qdrant_client.http import models as rest
from sentence_transformers import CrossEncoder

class RetrievalPipeline:
    def __init__(self, embedder: EmbeddingPipeline, store: VectorStore, reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.embedder = embedder
        self.store = store
        # Load cross-encoder for reranking
        print(f"Loading Cross-Encoder for Reranking: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model) 
        # In a real setup, we might use a multilingual reranker like BAAI/bge-reranker-v2-m3, 
        # but this is fine for demonstration and speed.

    def _reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """
        Fuses lists of results using Reciprocal Rank Fusion.
        """
        scores = {}
        for rank, res in enumerate(dense_results):
            if res.id not in scores:
                scores[res.id] = {"score": 0.0, "payload": res.payload, "dense_rank": rank, "sparse_rank": -1}
            scores[res.id]["score"] += 1.0 / (k + rank)
            
        for rank, res in enumerate(sparse_results):
            if res.id not in scores:
                scores[res.id] = {"score": 0.0, "payload": res.payload, "dense_rank": -1, "sparse_rank": rank}
            scores[res.id]["score"] += 1.0 / (k + rank)
            
        # Sort by fused score descending
        fused = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
        return [{"id": k, **v} for k, v in fused]
        
    def _calculate_confidence(self, reranked_results: List[Dict[str, Any]], fused_results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Calculates Retrieval Confidence based on scientific metrics.
        Returns label (HIGH, MEDIUM, LOW) and the reasoning.
        """
        if not reranked_results:
            return "LOW", {"reason": "No results returned."}
            
        top_score = reranked_results[0].get('rerank_score', 0.0)
        
        # Calculate gap between top 1 and top 2 if available
        gap = 0.0
        if len(reranked_results) > 1:
            gap = top_score - reranked_results[1].get('rerank_score', 0.0)
            
        # Agreement check: Did the top reranked result appear highly in both dense and sparse?
        top_id = reranked_results[0]['id']
        top_fused_meta = next((r for r in fused_results if r['id'] == top_id), None)
        
        dense_rank = top_fused_meta['dense_rank'] if top_fused_meta else -1
        sparse_rank = top_fused_meta['sparse_rank'] if top_fused_meta else -1
        
        agreement = (dense_rank != -1 and dense_rank < 10) and (sparse_rank != -1 and sparse_rank < 10)
        
        metrics = {
            "top_score": float(top_score),
            "score_gap": float(gap),
            "dense_sparse_agreement": agreement
        }
        
        # If reranker was skipped (indicated by top_score being a cosine similarity <= 1.0)
        # Cosine similarity thresholds: > 0.6 is HIGH, > 0.4 is MEDIUM, else LOW
        if top_score <= 1.0:
            if top_score > 0.6 and agreement:
                return "HIGH", metrics
            elif top_score > 0.4 or (top_score > 0.3 and gap > 0.1):
                return "MEDIUM", metrics
            else:
                return "LOW", metrics
                
        # Thresholds for Cross-Encoder (would be tuned in prod)
        if top_score > 3.0 and agreement:
            return "HIGH", metrics
        elif top_score > 0.0 or (top_score > -2.0 and gap > 1.0):
            return "MEDIUM", metrics
        else:
            return "LOW", metrics

    def retrieve(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full retrieval pipeline and tracks latency for each stage.
        """
        metrics = {}
        
        # 1. Query Embedding
        t0 = time.perf_counter()
        dense_q, sparse_q = self.embedder.embed_queries([query])
        dense_vec = dense_q[0]
        sparse_vec = sparse_q[0]
        metrics["query_embedding_ms"] = (time.perf_counter() - t0) * 1000
        
        top_k_retrieve = strategy.get("top_k_retrieve", 30)
        
        # 2. Dense Retrieval
        t0 = time.perf_counter()
        dense_results = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=dense_vec,
            using="dense",
            limit=top_k_retrieve,
        ).points
        metrics["dense_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        
        # 3. Sparse Retrieval
        t0 = time.perf_counter()
        sparse_indices = list(sparse_vec.keys())
        sparse_values = list(sparse_vec.values())
        sparse_results = self.store.client.query_points(
            collection_name=self.store.collection_name,
            query=rest.SparseVector(indices=sparse_indices, values=sparse_values),
            using="sparse",
            limit=top_k_retrieve
        ).points
        metrics["sparse_retrieval_ms"] = (time.perf_counter() - t0) * 1000
        
        # 4. Score Fusion (RRF)
        t0 = time.perf_counter()
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        metrics["rrf_fusion_ms"] = (time.perf_counter() - t0) * 1000
        
        # 5. Reranking
        t0 = time.perf_counter()
        top_k_rerank = strategy.get("top_k_rerank", 5)
        
        if top_k_rerank <= 0:
            final_results = fused[:top_k_retrieve]
            for r in final_results:
                # Find original dense score for absolute confidence
                dense_point = next((dr for dr in dense_results if dr.id == r['id']), None)
                r['rerank_score'] = float(dense_point.score) if dense_point else 0.0
            metrics["reranking_ms"] = (time.perf_counter() - t0) * 1000
        else:
            candidates = fused[:strategy.get("rerank_candidates", 20)] # Number of candidates to score
            
            if not candidates:
                return {"results": [], "confidence": "LOW", "confidence_metrics": {}, "latency_ms": metrics}
                
            cross_inp = [[query, c['payload']['text']] for c in candidates]
            scores = self.reranker.predict(cross_inp)
            
            for i in range(len(scores)):
                candidates[i]['rerank_score'] = float(scores[i])
                
            # Sort by rerank score descending
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            final_results = reranked[:top_k_rerank]
            metrics["reranking_ms"] = (time.perf_counter() - t0) * 1000
        
        # 6. Context Assembly & Confidence
        t0 = time.perf_counter()
        confidence, conf_metrics = self._calculate_confidence(final_results, fused)
        metrics["context_assembly_ms"] = (time.perf_counter() - t0) * 1000
        
        # Calculate total retrieval pipeline latency
        metrics["total_retrieval_ms"] = sum(metrics.values())
        
        return {
            "results": final_results,
            "confidence": confidence,
            "confidence_metrics": conf_metrics,
            "latency_ms": metrics
        }
