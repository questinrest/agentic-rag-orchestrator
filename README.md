# Agentic RAG Pipeline

A robust, stateful Retrieval-Augmented Generation (RAG) system synthesizing **Adaptive RAG**, **Corrective RAG (CRAG)**, and **Self-RAG** into a dynamic cyclical pipeline. Built using LangGraph, Pinecone, MongoDB, and Groq.

## Architecture

The system intelligently routes, validates, and refines queries using a rigorous, graph-based state machine that dynamically recovers from hallucinations and poor retrievals.

```mermaid
flowchart TD
    A([User Query]) --> CACHE{Semantic Cache?}
    CACHE -- Hit --> DONE([Return Cached Answer])
    CACHE -- Miss --> ROUTER{Adaptive Router}

    ROUTER -- llm_direct --> GEN[Generate]
    ROUTER -- web_search --> WEB[Web Search\nDuckDuckGo]
    ROUTER -- vectorstore --> RET

    subgraph RET["Retrieve Node"]
        direction LR
        PIN[Pinecone\nVector Search] --> RNK{Reranking\nEnabled?}
        RNK -- Yes --> RKR[Rerank Top-N] --> MDB[MongoDB\nParent Fetch]
        RNK -- No --> MDB
    end

    RET --> CRAG

    subgraph CRAG["CRAG Evaluate Node"]
        direction TB
        GR[Grade each doc\ncorrect / ambiguous / incorrect] --> ST[Extract relevant strips\nfrom correct + ambiguous docs]
        ST --> CHK{Any strips\nfound?}
    end

    CHK -- "Yes" --> GEN
    CHK -- "No (retries < 3)" --> RWR[Rewrite Query]
    CHK -- "No (retries ≥ 3)" --> WEB

    WEB --> GEN

    RWR -- "route = vectorstore" --> RET
    RWR -- "route = web_search" --> WEB

    GEN --> GRADE{grade_generation}

    GRADE -- "llm_direct\nOR retries ≥ 3\nOR halluc_retries ≥ 3" --> CACHE_STORE
    GRADE -- "route = web_search\nhallucination check\nbypassed → quality check only" --> QC{Answer\nUseful?}
    GRADE -- "route = vectorstore\nhallucination check runs" --> HALL{Grounded\nin context?}

    HALL -- "Yes" --> QC
    HALL -- "No (halluc_retries < 3)" --> REGEN[Regenerate\nbump halluc_retries]
    REGEN --> GEN

    QC -- "Yes" --> CACHE_STORE
    QC -- "No (retries < 3)" --> RWR

    CACHE_STORE[Update Semantic Cache] --> DONE
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