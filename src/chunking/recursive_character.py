import hashlib
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langsmith import traceable
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from .parent_child import compute_file_hash, document_loader


## splitter 
@traceable(run_type="chain", name="Recursive Character Splitter")
def splitter(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_records(chunks: List[Document], hash_val: str) -> List[Dict]:
    records = []
    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "_id": f"{hash_val}-chunk-{idx+1}",
                "chunk_text": chunk.page_content,
                "source": Path(chunk.metadata["source"]).name,
                "page": chunk.metadata.get("page"),
            }
        )
    return records


def ingest(file_path: Path):
    file_name = file_path.name
    hash_val = compute_file_hash(file_path=file_path)
    documents = document_loader(file_path=file_path)

    # chunks
    chunks = splitter(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    records = create_records(chunks=chunks, hash_val=hash_val)
    for r in records:
        r["source_hash_value"] = hash_val
        r["source"] = file_name

    return records