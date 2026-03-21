# TierRAG - Multi-Tiered Retrieval-Augmented Generation System

## Brief Description

**TierRAG** is a modular, high-performance Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **MongoDB Atlas**, and **Pinecone**. It is designed to handle user-authenticated document ingestion, multi-tier intelligent caching, and configurable chunking strategies.

Key capabilities include:
- **Multi-Tier Caching:** Implements an in-memory 3-tier cache (Exact, Semantic, and Retrieval) to aggressively minimize latency and LLM API costs.
- **Advanced Chunking:** Supports both simple Recursive Character chunking and relationship-preserving Parent-Child chunking.
- **Document Versioning:** Automatically manages active vs. archived versions of documents in MongoDB so overlapping document uploads don't pollute the vector space.
- **Namespace Isolation:** Enforces secure, per-user data isolation in Pinecone via JWT-based authentication.
- **Reranking & Generation:** Utilizes cross-encoder reranking (`bge-reranker-v2-m3`) and Groq's high-speed inference API (`llama-3.3-70b-versatile`) for highly accurate context generation.
- **Deep Observability:** Integrated with **LangSmith** to trace chunking, embedding, vector retrieval, and LLM generative logic at a granular level.

For a deep dive into the system's architecture, flow diagrams, and design decisions, see the [Architecture Document](docs/architecture.md).

---

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A MongoDB Atlas cluster (or local MongoDB string)
- A Pinecone account (API Key)
- A Groq account (API Key for LLM generation)

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd rag-project-2
```

### 3. Virtual Environment & Dependencies
Create and activate a virtual environment, then install the required Python packages:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Environment Configuration
Copy the `.env.example` file to create your own configuration file:
```bash
cp .env.example .env
```

Open `.env` and configure your credentials:
```env
# MongoDB Connection (Make sure to whitelist your IP in Atlas)
CONNECTION_STRING="mongodb+srv://<user>:<password>@cluster0...mongodb.net/"

# Auth Configuration
SECRET_KEY="your_secure_randomly_generated_jwt_secret"
ALGORITHM="HS256"

# Vector Database (Pinecone)
PINECONE_API_KEY="your_pinecone_api_key_here"

# LLM Generation (Groq)
GROQ_API_KEY="your_groq_api_key_here"

# Observability (LangSmith)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="TierRAG"
```

*(Note: Chunking parameters, cache thresholds, and model selections are further configurable in `src/config.py`)*

---

## How to Run

1. **Start the FastAPI Server:**
Ensure your virtual environment is activated, then run the application using Uvicorn:
```bash
uvicorn main:app --reload
```

2. **Access the API Documentation:**
Once the server is running, open your browser and navigate to the interactive Swagger UI:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

3. **General Usage Flow Testing:**
   - **Register & Login:** Use `POST /api/admin/register` then `POST /api/admin/login` to get a JWT Bearer token.
   - **Authorize Context:** Click the "Authorize" button in Swagger UI and paste the Bearer token.
   - **Upload Data:** Use `POST /upload` to upload a `.pdf` or `.txt` file.
   - **Query:** Use `POST /query` to ask questions about your uploaded documents and see the multi-tier caching in action!

---

## Project Structure

```
├── main.py                  # FastAPI app entrypoint
├── requirements.txt         # Project dependencies
├── api/
│   ├── auth/                # JWT Registration & login
│   ├── ingestion/           # Versioned document upload
│   └── generation/          # Query & Answer generation (Caching integration)
├── src/
│   ├── config.py            # Global Config (MongoDB, Pinecone, Chunking, Cache Thresholds)
│   ├── chunking/            # Splitting logic (Parent-Child & Recursive Character)
│   ├── embedding/           # Pinecone vectorization and indexing
│   ├── retrieval/           # Active doc filtering & Reranking
│   ├── generation/          # Groq LLM integration
│   └── caching/             # 3-Tier In-Memory Dictionary Caches (Exact, Semantic, Retrieval)
└── docs/
    └── architecture.md      # Detailed diagrams & design decisions
```