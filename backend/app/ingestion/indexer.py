"""
ingestion/indexer.py — ChromaDB Vector Indexing

Embeds document chunks and stores them in a persistent ChromaDB collection.
ChromaDB is the primary always-current vector store — updated on every upload.
FAISS/BM25 indexes in the retrieval modules are secondary and may lag until
their ingest() runs successfully.

Functions:
    index_chunks()            — embed and upsert chunks into ChromaDB
    search_chunks()           — semantic similarity search (used as fallback)
    delete_document_chunks()  — remove all chunks for a document on deletion
"""

from typing import List, Dict, Any
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from app.config.settings import settings

_chroma_client   = None
_collection      = None
_embedding_model = None
COLLECTION_NAME  = "documents"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        print(f"✅ ChromaDB client initialized → {settings.CHROMA_PERSIST_DIR}")
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ ChromaDB collection ready → '{COLLECTION_NAME}'")
    return _collection


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"⏳ Loading embedding model '{EMBEDDING_MODEL}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
    return _embedding_model


def index_chunks(
    chunks: List[Dict[str, Any]],
    document_id: str,
    uploaded_by: str = "",
    file_type: str = "",
) -> Dict[str, Any]:
    """Embed and upsert chunks into ChromaDB. Called on every document upload."""
    if not chunks:
        raise ValueError("No chunks to index")

    collection = get_collection()
    model      = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    print(f"⏳ Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    print(f"✅ Embeddings created")

    uploaded_at = datetime.utcnow().isoformat() + "Z"

    ids        = []
    documents  = []
    metadatas  = []

    for chunk, embedding in zip(chunks, embeddings):
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "chunk_id":    chunk["chunk_id"],
            "document_id": document_id,
            "source_name": chunk["source_name"],
            "page":        chunk.get("page", 1),
            "start_char":  chunk.get("start_char", 0),
            "end_char":    chunk.get("end_char", 0),
            "file_type":   file_type,
            "uploaded_by": uploaded_by,
            "uploaded_at": uploaded_at,
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"✅ {len(chunks)} chunks stored in ChromaDB")

    return {
        "chunks_stored": len(chunks),
        "collection":    COLLECTION_NAME,
        "document_id":   document_id,
        "metadata": {
            "file_type":   file_type,
            "uploaded_by": uploaded_by,
            "uploaded_at": uploaded_at,
        }
    }


def search_chunks(
    query: str,
    top_k: int = 5,
    document_ids: list = None,
) -> List[Dict[str, Any]]:
    """
    Semantic similarity search against ChromaDB.
    Used as the universal fallback when FAISS/BM25 indexes are unavailable.
    Always reflects the latest ingested documents.
    """
    collection = get_collection()
    model      = get_embedding_model()

    query_embedding = model.encode([query]).tolist()

    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    formatted = []
    if results["ids"] and results["ids"][0]:
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1
        ):
            score = round(1 - distance, 4)
            formatted.append({
                "chunk_id":    chunk_id,
                "document_id": metadata.get("document_id", ""),
                "source_name": metadata.get("source_name", ""),
                "text":        text,
                "score":       score,
                "rank":        rank,
                "metadata":    metadata,
            })

    return formatted


def delete_document_chunks(document_id: str) -> int:
    """Delete all ChromaDB chunks for a document. Called when a document is deleted."""
    collection = get_collection()
    results    = collection.get(where={"document_id": document_id})

    if not results["ids"]:
        return 0

    collection.delete(ids=results["ids"])
    print(f"🗑️ Deleted {len(results['ids'])} chunks for document {document_id}")
    return len(results["ids"])
