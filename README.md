# Agentic RAG Pipeline

A robust, stateful Retrieval-Augmented Generation (RAG) system synthesizing **Adaptive RAG**, **Corrective RAG (CRAG)**, and **Self-RAG** into a dynamic cyclical pipeline. Built using LangGraph, Pinecone, MongoDB, and Groq.

## Architecture

The system intelligently routes, validates, and refines queries using a rigorous, graph-based state machine that dynamically recovers from hallucinations and poor retrievals.

```mermaid
graph TD
    A([User Query]) --> Cache{Semantic Cache Hit?}
    Cache -- Yes --> K([Return Cached Answer])
    Cache -- No --> Router{Adaptive Router}

    %% --- Three Entry Routes ---
    Router -- vectorstore --> Retrieve[Retrieve Node]
    Router -- web_search --> WebSearch[Web Search Node]
    Router -- llm_direct --> Generate[Generate Node]

    %% --- Vectorstore Internal Detail ---
    subgraph Retrieve Node
        Pinecone[Pinecone Vector Search] --> RerankCheck{USE_RERANKING?}
        RerankCheck -- Yes --> Reranker[Serverless Rerank Top-N]
        RerankCheck -- No --> Mongo
        Reranker --> Mongo[Fetch Parent Docs - MongoDB]
    end

    %% --- CRAG Evaluate ---
    Retrieve --> Evaluate{CRAG Document Grader}
    Evaluate -- Relevant --> Generate
    Evaluate -- "Irrelevant & retries < 3" --> RewriteQuery[Rewrite Query Node]
    Evaluate -- "Irrelevant & retries >= 3" --> WebSearch

    %% --- Web Search & Rewrite Convergence ---
    WebSearch --> Generate
    RewriteQuery --> Retrieve

    %% --- grade_generation conditional edge ---
    Generate --> GradeGen{"grade_generation()"}

    GradeGen -- "llm_direct route\nOR retries >= 3" --> CacheStore[Update Semantic Cache]
    GradeGen -- "Hallucination: no\n(vectorstore only)" --> Generate
    GradeGen -- "Hallucination: yes\nQuality: no" --> RewriteQuery
    GradeGen -- "Hallucination: yes\nQuality: yes" --> CacheStore

    CacheStore --> K
```

## framework choice & why

- **langgraph**: used as an orchestration engine for the agentic rag pipeline. 
- **groq**: offers free tier for llms.
- **duckduckgo search**: free search api. acts as a fallback.
- **mongodb + pinecone**: hybrid data layer. pinecone is fast for vector search. mongodb holds the heavy parent chunks with relevant metadata.

## chunking strategy & why

i used a **parent-child chunking** method.
- **the process**: split docs into big parent chunks. then split those into small child chunks. pinecone only stores the small child chunks. if pinecone finds a match, it gives me the `parent_id`, used this to fetch the huge parent text from mongodb.
- **why?**: this gives me the best of both worlds. tiny chunks give exact search hits. big chunks give the llm rich context to answer correctly.

## design trade-offs

1. **latency vs. accuracy**
running multiple checks adds latency. but it reduces hallucinations.

2. **deterministic loop boundaries**
self-rag loops can run forever. it might keep searching for missing facts. i added a hard limit. the system tries max 3 times.


## evaluation (ragas)

because i don't have access to an explicitly labeled ground-truth dataset, i built an automated evaluation pipeline using the **ragas** framework. it focuses on two critical reference-free metrics to grade the language model:
- **faithfulness**: checks if the llm hallucinated facts outside the source documents.
- **answer relevancy**: checks if the generated answer directly addresses the user's intent.

### test cases & results

i evaluated 3 distinct scenarios representing the pipeline's core branches:

1. **vectorstore route**: *"Tell me about the cross-modal architecture of RAG-Anything."*
   - **faithfulness**: `__` (highly grounded in the retrieved child chunks)
   - **answer relevancy**: `__` (focused entirely on the architecture components)

2. **web search fallback**: *"Who won the latest Super Bowl?"*
   - **faithfulness**: `__` (copied directly from duckduckgo snippet)
   - **answer relevancy**: `__` (direct factual answer)

3. **direct llm route**: *"What is the capital of France?"*
   - **faithfulness**: `__` (bypasses retrieval, relies on parametric knowledge)
   - **answer relevancy**: `__` (direct factual answer)

> **note**: you can reproduce these metrics by running the `notebooks/ragas_evaluation.ipynb` file locally. it uses custom huggingface embeddings and groq to prevent api costs!

## Quick Start Demo

Run the fully unified Agentic pipeline interactively from your terminal to see the routing branches natively executed:

```bash
# 1. Start the Virtual Environment
.venv\Scripts\activate

# 2. Run the Demo CLI
python main.py --query "Tell me about RAG-Anything's architecture"
```

For Jupyter fans, refer to `notebooks/demo_agentic_rag.ipynb` to view step-by-step stream traces of the Graph navigating edge constraints.