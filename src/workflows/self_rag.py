from langchain_groq import ChatGroq
from src.config import SELF_RAG_MODEL, SELF_RAG_MAX_TOKENS, SELF_RAG_API_KEY, SELF_RAG_TEMPERATURE
from src.workflows.state import AgenticRAGState

__all__ = ["rewrite_query_node", "hallucination_grader", "answer_quality_grader"]

llm_evaluator = ChatGroq(
    model=SELF_RAG_MODEL,
    api_key=SELF_RAG_API_KEY,
    temperature=SELF_RAG_TEMPERATURE,
    max_tokens=SELF_RAG_MAX_TOKENS,
)

def rewrite_query_node(state: AgenticRAGState):
    query = state["query"]
    prompt = f"""
    Look at the input and try to reason about the underlying semantic intent / meaning.
    Here is the initial question:
    -------
    {query}
    -------
    Formulate an improved question:
    """
    response = llm_evaluator.invoke(prompt).content.strip()
    return {"query": response, "retries": state.get("retries", 0) + 1}


def hallucination_grader(state: AgenticRAGState) -> str:
    """Checks if the answer is grounded in the documents. Returns 'yes' or 'no'."""
    answer = state.get("answer", "")
    context = "\n".join(state.get("final_context_strips", []))
    if not context.strip() or state.get("route") != "vectorstore":
        return "yes"
        
    prompt = f"""
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.
    
    Context: {context}
    
    Answer: {answer}
    
    Binary score (yes or no):
    """
    response = llm_evaluator.invoke(prompt).content.strip().lower()
    return "yes" if "yes" in response else "no"


def answer_quality_grader(state: AgenticRAGState) -> str:
    """Checks if the answer resolves the query. Returns 'yes' or 'no'."""
    answer = state.get("answer", "")
    query = state.get("original_query", state["query"])
    prompt = f"""
    You are a grader assessing whether an answer is useful to resolve a question.
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    
    Question: {query}
    
    Answer: {answer}
    
    Binary score (yes or no):
    """
    response = llm_evaluator.invoke(prompt).content.strip().lower()
    return "yes" if "yes" in response else "no"
