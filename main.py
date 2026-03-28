import sys
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


from langgraph.graph import StateGraph, START, END

from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.workflows.state import AgenticRAGState
from src.adaptive_router import llm_router
from src.workflows.crag import retrieve_node, evaluate_node, web_search_node, generate_node
from src.workflows.self_rag import rewrite_query_node, hallucination_grader, answer_quality_grader

def router_node(state: AgenticRAGState):
    route = llm_router(state["query"])
    if "web_search" in route.lower():
        route = "web_search"
    elif "llm_direct" in route.lower():
        route = "llm_direct"
    else:
        route = "vectorstore"
    return {"route": route, "original_query": state.get("original_query") or state.get("query")}

def route_start(state: AgenticRAGState) -> str:
    return state["route"]

def rewrite_routing(state: AgenticRAGState) -> str:
    """After a query rewrite, route back to web_search if that was the original route.
    Otherwise route back to vector retrieve."""
    if state.get("route") == "web_search":
        logger.info("Rewrite routing: original route was web_search → retrying web search")
        return "web_search"
    logger.info("Rewrite routing: original route was vectorstore → retrying retrieve")
    return "retrieve"

def evaluate_cond_edge(state: AgenticRAGState) -> str:
    if state["needs_web_search"]:
        if state.get("retries", 0) < 3:
            return "rewrite_query"
        else:
            return "web_search"
    return "generate"

def grade_generation(state: AgenticRAGState) -> str:
    if state.get("retries", 0) >= 3:
        return END

    if state["route"] == "llm_direct":
        return END

    hallucination_score = hallucination_grader(state)
    if hallucination_score == "no":
        return "generate" 
        
    quality_score = answer_quality_grader(state)
    if quality_score == "no":
        return "rewrite_query" 
        
    return END

# Build Full Agentic RAG Graph
builder = StateGraph(AgenticRAGState)

builder.add_node("router", router_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("web_search", web_search_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_start,
    {
        "vectorstore": "retrieve",
        "web_search": "web_search",
        "llm_direct": "generate"
    }
)

builder.add_edge("retrieve", "evaluate")

builder.add_conditional_edges(
    "evaluate",
    evaluate_cond_edge,
    {
        "rewrite_query": "rewrite_query",
        "web_search": "web_search",
        "generate": "generate"
    }
)

builder.add_conditional_edges(
    "rewrite_query",
    rewrite_routing,
    {
        "web_search": "web_search",
        "retrieve": "retrieve"
    }
)
builder.add_edge("web_search", "generate")

builder.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "generate": "generate", # Loop to regenerate if hallucinates
        "rewrite_query": "rewrite_query", # Loop to search again if no useful answer
        END: END
    }
)

agentic_rag_app = builder.compile()


from src.caching.semantic_cache import get_semantic_cache, set_semantic_cache

def run_agentic_rag(query: str, namespace: str = "default_namespace") -> str:
    """Main Orchestrator Entrypoint"""
    
    # 1. Check Semantic Cache First
    cached_answer, cached_sources, query_emb = get_semantic_cache(query=query)
    if cached_answer:
        logger.info("Semantic Cache Hit")
        return cached_answer
        
    initial_state = {
        "query": query,
        "original_query": query,
        "namespace": namespace,
        "documents": [],
        "final_context_strips": [],
        "needs_web_search": False,
        "answer": "",
        "retries": 0,
        "route": ""
    }
    
    # 2. Execute Graph
    result = agentic_rag_app.invoke(initial_state)
    final_answer = result.get("answer", "I don't have enough information to answer that.")
    
    # 3. Store in Semantic Cache if valid
    if final_answer and final_answer != "I don't have enough information to answer that.":
        # We can extract sources from final_context_strips or just leave empty if it's direct LLM
        sources = result.get("final_context_strips", [])
        set_semantic_cache(query=query, answer=final_answer, sources=sources, query_emb=query_emb)
        
    return final_answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agentic RAG CLI Demo")
    parser.add_argument("--query", type=str, default="Tell me about cross-modal graphs", help="Query to run")
    args = parser.parse_args()
    
    print(f"\\n[User Query]: {args.query}\\n")
    logger.info("Running Agentic RAG Pipeline...")
    answer = run_agentic_rag(args.query)
    logger.info("Agentic RAG Pipeline Execution Completed")
    print("\\n[Final Response]:\\n", answer)
