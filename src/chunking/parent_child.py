import hashlib
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langsmith import traceable
from src.config import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)



# Compute file hash


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()



# Document Loader

def document_loader(file_path: Path) -> List[Document]:
    file_format = file_path.suffix.lower()
    if file_format == ".pdf":
        loader = PyMuPDFLoader(file_path=str(file_path))
    elif file_format == ".txt":
        loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported document format: {file_format}")
    return loader.load()


# Parent Splitter

@traceable(run_type="chain", name="Parent Splitter")
def parent_splitter(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    parent_chunks = splitter.split_documents(documents)
    return parent_chunks



# Child Splitter

@traceable(run_type="chain", name="Child Splitter")
def children_splitter(
    parent_chunks: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    child_chunks = splitter.split_documents(parent_chunks)
    return child_chunks


# Parent -> Child Mapping

def create_parent_child_mapping(parent_chunks: List[Document], hash_val: str):
    for idx, chunk in enumerate(parent_chunks):
        chunk.metadata["parent_id"] = f"{hash_val}-parent-{idx+1}"
    child_chunks = children_splitter(
        parent_chunks,
        CHILD_CHUNK_SIZE,
        CHILD_CHUNK_OVERLAP
    )
    return parent_chunks, child_chunks



# Parent Store (Dictionary)

def parent_store(parent_chunks: List[Document]) -> Dict[str, str]:
    store = {}
    for idx, pchunk in enumerate(parent_chunks):
        store[pchunk.metadata["parent_id"]] = pchunk.page_content
    return store


# Create Child Records

def create_child_records(child_chunks: List[Document], hash_val: str) -> List[Dict]:
    records = []
    for idx, chunk in enumerate(child_chunks):
        records.append(
            {
                "_id": f"{hash_val}-chunk-{idx+1}",
                "chunk_text": chunk.page_content,
                "parent_id": chunk.metadata["parent_id"],
                "source": Path(chunk.metadata["source"]).name,
                "page": chunk.metadata.get("page"),
            }
        )
    return records


# Ingest Function

def ingest(file_path: Path):
    file_name = file_path.name
    hash_val = compute_file_hash(file_path=file_path)
    documents = document_loader(file_path=file_path)

    # parent chunks
    parent_chunks = parent_splitter(
        documents,
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )

    parent_chunks, child_chunks = create_parent_child_mapping(parent_chunks, hash_val)
    parents = parent_store(parent_chunks)
    records = create_child_records(child_chunks, hash_val)
    for r in records:
        r["source_hash_value"] = hash_val
        r["source"] = file_name

    return records, parents