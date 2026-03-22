# Agentic RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) system built with LangChain, Pinecone, MongoDB, and Groq. It features intelligent routing, context retrieval using Parent-Child chunking, and semantic caching. Advanced workflows are currently explored in Jupyter Notebooks.

## Key Features (Core)

- **Adaptive Routing**: Routes queries to Vectorstore, Direct LLM, or Web Search via an LLM classifier.
- **Semantic Caching**: Uses `sentence-transformers` for in-memory caching to avoid redundant API calls.
- **Parent-Child Chunking**: Retrieves narrow chunks from Pinecone but feeds broader parent context from MongoDB to the LLM. 
- **Vector Search**: Uses Pinecone for storage and search, with cross-encoder reranking.
- **Fast Generation**: Uses Groq API for high-speed, reliable generation.

## Learning & Exploration (Notebooks)

I am actively experimenting with LangGraph and Corrective RAG (CRAG) in the `notebooks/` directory. These features will be integrated into the core codebase later:
- Document grading
- Query rewriting
- Web search fallbacks
- LangGraph state loops

## Project Status

*(Items marked "Exploration" are prototyped in notebooks, pending core integration).*

- [x] **Ingest documents**: Load ≥ 3 docs, chunk, embed, store in DB.
- [x] **Query Router**: Classify query as llm_direct, vectorstore, or web_search.
- [x] **Retrieve**: Fetch top-k chunks from vector store.
- [ ] **Grade Documents**: Binary relevance score via LLM. *(Exploration)*
- [ ] **Query Rewriter**: Rewrite query if chunks are irrelevant. *(Exploration)*
- [ ] **Fallback**: Use web search or LLM knowledge if retrieval fails. *(Exploration)*
- [x] **Generate Answer**: Synthesise context into a grounded response.
- [ ] **Hallucination Grader**: Check if answer is supported by sources. *(Pending)*
- [ ] **Answer Quality Grader**: Check if answer addresses the question. *(Pending)*
- [ ] **Loop Control**: Max 3 retries, terminate gracefully. *(Exploration)*
- [x] **Demo**: Notebooks showing pipeline branches.

## Architecture

1. **Ingestion (`src/chunking`)**: Documents are split into parent-child chunks and stored in MongoDB.
2. **Embedding (`src/embedding`)**: Child chunks are embedded and upserted into Pinecone.
3. **Routing (`src/adaptive_router.py`)**: Evaluates query to decide the appropriate handler.
4. **Caching (`src/caching`)**: Semantically similar queries return cached results instantly.
5. **Retrieval (`src/retrieval`)**: Embedded queries match against Pinecone and get reranked.
6. **Generation (`src/generation`)**: The LLM aggregates parent chunks and generates a response.

## Setup Instructions

### 1. Prerequisites
- Python 3.11+
- MongoDB Atlas
- Pinecone API Key
- Groq API Key

### 2. Installation
```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Configuration
Set up your `.env`:
```env
CONNECTION_STRING="mongodb+srv://<user>:<password>@cluster0...mongodb.net/"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_INDEX_NAME="agenticrag"
GROQ_API_KEY="your_groq_api_key_here"
OPENAI_MODEL_GROQ="openai/gpt-oss-120b"
```

## Project Structure

```text
├── src/
│   ├── config.py                 # Global configurations
│   ├── adaptive_router.py        # LLM query router
│   ├── chunking/                 # Document ingestion & splitting
│   ├── embedding/                # Pinecone upsert logic
│   ├── retrieval/                # Pinecone vector search & reranking
│   ├── generation/               # LLM generation
│   └── caching/                  # Semantic cache
├── notebooks/                    # Exploration and prototyping
├── .env.example                  # Example environment variables
└── requirements.txt              # Dependencies
```