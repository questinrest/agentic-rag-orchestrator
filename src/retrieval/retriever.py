from typing import List, Dict
from src.embedding.embed import INDEX
from src.config import TOP_K
from src.utils.logger import get_logger

logger = get_logger(__name__)

def search_vector_db(
    namespace: str,
    query: str,
    top_k: int = TOP_K
) -> List[Dict]:

    results = INDEX.search(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": top_k
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

    logger.info(f"Search vector DB returned {len(retrieved)} hits for namespace {namespace}")
    return retrieved