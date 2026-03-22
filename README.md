# Agentic RAG Pipeline

A highly modular and efficient Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Pinecone**, **MongoDB**, and **Groq for LLM inferencing**. This project focuses on intelligent routing, optimized context retrieval using Parent-Child chunking, semantic caching, and full observability. Advanced agentic workflows (like Corrective RAG) are currently being explored and prototyped in Jupyter Notebooks before integration into the core codebase.

## Key Features (Core Codebase)

- **Adaptive Routing**: Intelligently routes user queries to the appropriate handler (Vectorstore, Direct LLM, or Web Search) using a Groq-powered classifier (`adaptive_router.py`).
- **Semantic Caching**: Implements an in-memory semantic cache using `sentence-transformers` (`all-MiniLM-L6-v2`). This avoids redundant embedding and LLM calls for semantically similar queries by returning cached answers based on a cosine similarity threshold.
- **Advanced Chunking Strategy**: Supports `parent_child`. By utilizing Parent-Child mapping, the system retrieves narrow, highly relevant chunks (children) from Pinecone but feeds the LLM the broader context (parents) from MongoDB to maintain full semantic meaning without token bloat.
- **High-Performance Vector Search & Reranking**: Uses Pinecone for vector storage and search, supporting cross-encoder reranking (`bge-reranker-v2-m3`) to surface the most relevant chunks.
- **High-Speed Generation**: Integrates the blazing-fast Groq API (`openai/gpt-oss-120b` or similar) to generate final answers reliably with precise citations.

## Learning & Exploration (Notebooks)

We are actively experimenting with **LangGraph** and **Corrective RAG (CRAG)** workflows to make the system fully self-correcting. The following features are currently prototyped in the `notebooks/` directory and are slated for core integration:
- Document Grading (binary relevance scoring)
- Query Rewriting (looping back when retrieval fails)
- Web Search Fallbacks
- LangGraph continuous state control loops

## Current Project Status (Agentic Pipeline Checklist)

Here is the progress on the core Agentic Pipeline goals. *(Note: Items marked as "Exploration" are currently prototyped in notebooks, waiting to be modularized into `src/`)*:

- [x] **Ingest documents** - load ≥ 3 docs, chunk, embed, store in a vector DB
- [x] **Query Router** - classify query as llm_direct / vectorstore / web_search using structured LLM output
- [x] **Retrieve** - fetch top-k chunks from vector store
- [ ] **Grade Documents** - binary relevance score (yes/no) per chunk via LLM *(Exploration)*
- [ ] **Query Rewriter** - if all chunks irrelevant, rewrite the query and retry *(Exploration)*
- [ ] **Fallback** - if retry fails, use web search (Tavily) or LLM parametric knowledge *(Exploration)*
- [x] **Generate Answer** - synthesise context into a grounded response
- [ ] **Hallucination Grader** - check if answer is supported by source documents *(Pending)*
- [ ] **Answer Quality Grader** - check if answer actually addresses the question *(Pending)*
- [ ] **Loop Control** - max 3 retry iterations, system must terminate gracefully *(Exploration)*
- [x] **Demo** - notebooks showing all pipeline branches in action

## Architecture

1. **Document Ingestion & Chunking (`src/chunking/parent_child.py`)**: Documents (PDF/TXT) are hashed, loaded, and split. If using Parent-Child chunking, larger parent chunks are stored in MongoDB while smaller child chunks are mapped to their parents.
2. **Embedding & Upsert (`src/embedding/embed.py`)**: Text chunks are embedded and upserted into an isolated Pinecone namespace.
3. **Query Routing (`src/adaptive_router.py`)**: A routing classifier evaluates the query to decide if it requires RAG, web search, or just general knowledge.
4. **Caching (`src/caching/semantic_cache.py`)**: The query's embedding is checked against a semantic cache. On a hit, the cached answer is returned immediately.
5. **Retrieval (`src/retrieval/reranker.py`, `retriever.py`)**: If a vector search is needed, queries are embedded and matched against Pinecone, with optional reranking.
6. **Generation (`src/generation/generator.py`)**: Retrieved child chunks trigger a lookup of their corresponding parent chunks from MongoDB. The LLM aggregates this full context and generates a well-cited response.

## Setup Instructions

### 1. Prerequisites
- Python 3.11+
- A MongoDB Atlas cluster
- Pinecone Account & API Key
- Groq API Key

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 3. Create a Virtual Environment and Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configuration
Copy the sample exact configuration to `.env`:
```bash
cp .env.example .env
```

Set up your `.env` variables:
```env
# MongoDB Connection
CONNECTION_STRING="mongodb+srv://<user>:<password>@cluster0...mongodb.net/"

# Vector Database (Pinecone)
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="agenticrag"

# LLM Generation (Groq)
GROQ_API_KEY="your_groq_api_key_here"
OPENAI_MODEL_GROQ="openai/gpt-oss-120b"
```

You can further tweak chunk sizes, overlaps, caching thresholds (`SEMANTIC_CACHE_THRESHOLD`), and models inside `src/config.py`.

## Project Structure

```text
├── src/
│   ├── config.py                 # Global configurations (Model names, chunk sizes, DB names)
│   ├── adaptive_router.py        # LLM-based query router
│   ├── chunking/
│   │   └── parent_child.py       # Logic for parent-child split and document loading
│   ├── embedding/
│   │   └── embed.py              # Pinecone index management and batch upsert
│   ├── retrieval/
│   │   ├── retriever.py          # Standard Pinecone vector search
│   │   └── reranker.py           # Pinecone search with Cross-Encoder reranking
│   ├── generation/
│   │   └── generator.py          # LLM answer generation and context building
│   └── caching/
│       └── semantic_cache.py     # In-memory scalable semantic cache
├── notebooks/                    # Jupyter notebooks for testing and experimentation
├── .env.example                  # Example environment variables
└── requirements.txt              # Project dependencies
```