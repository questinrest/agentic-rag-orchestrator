from langchain_groq import ChatGroq

from src.config import (
    ADAPTIVE_RAG_MODEL,
    ADAPTIVE_RAG_TEMPERATURE,
    ADAPTIVE_RAG_MAX_TOKENS,
    ADAPTIVE_RAG_API_KEY,
)

llm_for_routing = ChatGroq(
    model=ADAPTIVE_RAG_MODEL,
    api_key=ADAPTIVE_RAG_API_KEY,
    temperature=ADAPTIVE_RAG_TEMPERATURE,
    max_tokens=ADAPTIVE_RAG_MAX_TOKENS,
)


def llm_router(query: str) -> str:
    prompt = f"""
    You are a strict routing classifier for a RAG system.

Decide the best route for answering the query:

ROUTES:
- llm_direct: Use ONLY if the question can be answered from general knowledge and does NOT require any specific documents or private data.
- vectorstore: Use if the query likely depends on documents, internal knowledge base, PDFs, embeddings, or RAG-related information.
- web_search: Use if the query requires real-time or recent information (news, prices, current events).

IMPORTANT RULES:
- If unsure between llm_direct and vectorstore then you can choose vectorstore
- If the query mentions documents, context, knowledge base, embeddings, or "RAG" then choose vectorstore
- Be deterministic. Output ONLY one word from the routes.

Query: {query}

Answer:
    """
    return llm_for_routing.invoke(prompt).content.strip()
