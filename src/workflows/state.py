from typing import List, TypedDict

class AgenticRAGState(TypedDict):
    """
    State for the Agentic RAG Pipeline (Adaptive + CRAG + Self-RAG).
    """
    query: str
    original_query: str
    namespace: str
    documents: List[str]
    final_context_strips: List[str]
    needs_web_search: bool
    answer: str
    retries: int
    hallucination_retries: int
    route: str
