import os
from pathlib import Path

from dotenv import load_dotenv

try:
    import certifi
except ModuleNotFoundError:
    certifi = None

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None


load_dotenv()


# MongoDB
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
DATABASE_NAME = "AgenticRAG"

client = None
db = None
parent_store_collection = None

if MongoClient is not None and CONNECTION_STRING:
    tls_ca_file = certifi.where() if certifi is not None else None
    client = MongoClient(CONNECTION_STRING, tlsCAFile=tls_ca_file)
    db = client[DATABASE_NAME]
    parent_store_collection = db["parent_store"]


# Pinecone
PINECONE_API_KEY : str = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD : str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION : str = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_NAME : str = os.getenv("PINECONE_INDEX_NAME", "agenticrag")
PINECONE_EMBEDDING_MODEL : str = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
PINECONE_RERANKER_MODEL : str = os.getenv("PINECONE_RERANKER_MODEL", "bge-reranker-v2-m3")
BATCH_SIZE : int = int(os.getenv("BATCH_SIZE", 96))

# Retrieval
TOP_K : int = int(os.getenv("TOP_K", 10))
TOP_N : int = int(os.getenv("TOP_N", 5))


# Chunking Strategy
# choose from : "recursive_character", "parent_child"
CHUNKING_STRATEGY : str = os.getenv("CHUNKING_STRATEGY", "parent_child")

# Parent Child Chunking
PARENT_CHUNK_SIZE : int = int(os.getenv("PARENT_CHUNK_SIZE", 1000))
PARENT_CHUNK_OVERLAP : int = int(os.getenv("PARENT_CHUNK_OVERLAP", 200))
CHILD_CHUNK_SIZE : int = int(os.getenv("CHILD_CHUNK_SIZE", 200))
CHILD_CHUNK_OVERLAP : int = int(os.getenv("CHILD_CHUNK_OVERLAP", 20))

# LLM (Groq)
GROQ_API_KEY : str = os.getenv("GROQ_API_KEY")
OPENAI_MODEL_GROQ : str = os.getenv("OPENAI_MODEL_GROQ", "openai/gpt-oss-120b")
TEMPERATURE : float = float(os.getenv("TEMPERATURE", 0.2))
MAX_TOKENS : int = int(os.getenv("MAX_TOKENS", 1024))


# CACHE Settings
SEMANTIC_CACHE_THRESHOLD : float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
