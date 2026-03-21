from typing import List, Dict
from src.embedding.embed import INDEX
from src.config import TOP_K, TOP_N, PINECONE_RERANKER_MODEL
from langsmith import traceable


@traceable(run_type="retriever", name="Pinecone Retrieval with Reranker")
def search_vector_db_reranker(
    namespace: str,
    query: str,
    top_k: int = TOP_K,
    top_n: int = TOP_N,
    rerank_model: str = PINECONE_RERANKER_MODEL
) -> List[Dict]:

    results = INDEX.search(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
        },
        rerank={
            "model": rerank_model,
            "top_n": top_n,
            "rank_fields": ["chunk_text"]
        },
        fields=["source", "chunk_text", "page", "parent_id"]
    )

    hits = results.get("result", {}).get("hits", [])

    retrieved = []
    for hit in hits:
        fields = hit.get("fields", {})
        retrieved.append({
            "id": hit.get("_id", ""),
            "score": hit.get("_score", 0),
            "chunk_text": fields.get("chunk_text", ""),
            "page": fields.get("page", ""),
            "source": fields.get("source", ""),
            "parent_id": fields.get("parent_id", ""),
        })

    return retrieved