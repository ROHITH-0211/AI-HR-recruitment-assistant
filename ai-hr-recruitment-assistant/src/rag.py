"""
RAG (Retrieval-Augmented Generation) Module
--------------------------------------------
Architecture:

    Knowledge Base Documents (data/knowledge_base/*.txt)
            |
    Document Chunking
            |
    Embeddings (Sentence Transformers)
            |
    ChromaDB (vector store)
            |
    Retriever
            |
    Relevant Recruitment Knowledge
            |
    OpenRouter LLM

Builds a small vector knowledge base from text files in
data/knowledge_base/ (interview guidelines, skill definitions, recruitment
best practices, job role requirements) and retrieves the most relevant
chunks for a given query. Retrieval only happens when a caller actually
asks for it (e.g. while generating interview questions) — it is not
injected into every LLM call.
"""

import glob
import os
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_PERSIST_DIR, KNOWLEDGE_BASE_DIR, EMBEDDING_MODEL

COLLECTION_NAME = "recruitment_knowledge_base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base directory has no usable documents."""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunker so long knowledge-base docs retrieve well."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def get_chroma_collection():
    """Get (or create) the persistent Chroma collection for the knowledge base."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def build_knowledge_base(force_rebuild: bool = False) -> int:
    """
    Ingest all .txt/.md files from data/knowledge_base/ into the vector store.
    Returns the number of chunks indexed. Idempotent unless force_rebuild=True.
    """
    collection = get_chroma_collection()

    if force_rebuild:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    if collection.count() > 0 and not force_rebuild:
        return collection.count()

    files = (
        glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))
        + glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))
    )
    if not files:
        raise KnowledgeBaseError(
            f"No knowledge base documents found in '{KNOWLEDGE_BASE_DIR}/'. "
            "Add at least one .txt or .md file there."
        )

    ids, documents, metadatas = [], [], []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        source_name = os.path.basename(filepath)
        for i, chunk in enumerate(_chunk_text(content)):
            ids.append(f"{source_name}-{i}")
            documents.append(chunk)
            metadatas.append({"source": source_name})

    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection.count()


def retrieve_knowledge(query: str, n_results: int = 3) -> List[dict]:
    """Retrieve the top-N most relevant knowledge base chunks for a query."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        build_knowledge_base()

    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(documents, metadatas):
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})
    return chunks
