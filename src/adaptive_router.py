from langchain_groq import ChatGroq
from langsmith import traceable
from src.config import (
    OPENAI_MODEL_GROQ,
    TEMPERATURE,
    MAX_TOKENS,
    GROQ_API_KEY,
)

llm_for_routing = ChatGroq(
    model=OPENAI_MODEL_GROQ,
    api_key=GROQ_API_KEY,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

@traceable(name="Adaptive Router")
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
