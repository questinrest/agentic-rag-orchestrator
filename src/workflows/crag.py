import re
from typing import List, Dict, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

from src.retrieval.retriever import search_vector_db
from src.generation.generator import llm
from src.config import parent_store_collection
from src.config import CRAG_MODEL, CRAG_MAX_TOKENS, CRAG_API_KEY, CRAG_TEMPERATURE
from src.workflows.state import AgenticRAGState

llm_evaluator = ChatGroq(
    model=CRAG_MODEL,
    api_key=CRAG_API_KEY,
    temperature=CRAG_TEMPERATURE,
    max_tokens=CRAG_MAX_TOKENS,
)


def clean_and_split(text: str) -> List[str]:
    text = re.sub(r'-\s*\n\s*', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def crag_router(query: str, doc: str) -> str:
    prompt = f"""
    You are a strict document evaluator for a RAG system.
    You're task is to rate each query and doc pair and tell if it is "correct", "incorrect", or "ambiguous".
    "correct" means the document is highly relevant to the query.
    "incorrect" means the document is not relevant to the query.
    "ambiguous" means the document is partially relevant or you are unsure.
    Answer only "correct", "incorrect", or "ambiguous".

Query : {query}
Doc : {doc}

Answer:
    """
    return llm_evaluator.invoke(prompt).content.strip().lower()


def retrieve_node(state: AgenticRAGState):
    query = state["query"]
    namespace = state.get("namespace", "default_namespace")
    
    retrieved_chunks = search_vector_db(
        namespace=namespace, 
        query=query, 
        top_k=3
    )
    
    parent_ids = {chunk.get("parent_id") for chunk in retrieved_chunks if chunk.get("parent_id")}
    
    documents = []
    if parent_ids:
        parent_docs = list(parent_store_collection.find({
            "parent_id": {"$in": list(parent_ids)},
            "namespace": namespace
        }))
        for doc in parent_docs:
            documents.append(doc["text"])
    else:
        for chunk in retrieved_chunks:
            documents.append(chunk.get("chunk_text", ""))
            
    return {"documents": documents}


def evaluate_node(state: AgenticRAGState):
    query = state["query"]
    documents = state["documents"]
    
    final_context_strips = []
    needs_web_search = False
    
    for doc in documents:
        eval_result = crag_router(query, doc)
        
        if "correct" in eval_result:
            strips = clean_and_split(doc)
            for strip in strips:
                strip_eval = crag_router(query, strip)
                if strip_eval in ["correct", "ambiguous", "good"]:
                    final_context_strips.append(strip)
                    
        elif "ambiguous" in eval_result:
            needs_web_search = True
            strips = clean_and_split(doc)
            for strip in strips:
                strip_eval = crag_router(query, strip)
                if strip_eval in ["correct", "ambiguous", "good"]:
                    final_context_strips.append(strip)
                    
        else: # incorrect
            needs_web_search = True

    if len(final_context_strips) == 0:
        needs_web_search = True
            
    return {"final_context_strips": final_context_strips, "needs_web_search": needs_web_search}


def web_search_node(state: AgenticRAGState):
    query = state["query"]
    final_context_strips = state.get("final_context_strips", [])
    
    search = DuckDuckGoSearchRun()
    try:
        results = search.invoke(query)
        final_context_strips.append(f"[Web Search Results]: {results}")
    except Exception as e:
        print(f"Web search failed: {e}")
        
    return {"final_context_strips": final_context_strips}


def generate_node(state: AgenticRAGState):
    query = state["query"]
    # If there are context strips, use them. Else, just use the query (LLM direct logic).
    final_context = "\n".join(state.get("final_context_strips", []))
    
    if final_context.strip():
        messages = [
            ("system", "You are an assistant. Answer the query based on the context provided. If there's no useful context, state that explicitly."),
            ("human", f"Context:\n{final_context}\n\n---\nQuestion: {query}")
        ]
    else:
        messages = [
            ("system", "You are a helpful assistant. Provide an answer from your general knowledge."),
            ("human", f"Question: {query}")
        ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}
