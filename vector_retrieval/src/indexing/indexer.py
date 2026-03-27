"""
src/indexing/indexer.py
Orchestrates the full indexing pipeline:
  load document → chunk → embed → store in vector index.
"""

from src.utils.loader import load_document
from src.utils.chunker import chunk_text_with_metadata
from src.models.embedding_model import load_embedding_model, encode_chunks
from src.indexing.vector_store import build_index
from typing import Tuple, List, Dict, Any
import faiss


def build_pipeline(
    document_source: str,
    document_title: str,
    document_id: str,
    chunk_size: int = 150,
    chunk_overlap: int = 30,
    model_name: str = 'all-MiniLM-L6-v2'
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]], object]:
    """
    Run the full indexing pipeline for a single document.

    Steps:
        1. Load document from disk
        2. Chunk into overlapping word windows with metadata
        3. Encode chunks into embedding vectors
        4. Build and return a FAISS index

    Args:
        document_source: File path to the document.
        document_title:  Human-readable title of the document.
        document_id:     Unique identifier for the document.
        chunk_size:      Number of words per chunk.
        chunk_overlap:   Number of overlapping words between chunks.
        model_name:      SentenceTransformer model to use for embeddings.

    Returns:
        Tuple of (faiss_index, chunk_records, embedding_model)
    """
    print('Loading document...')
    document_text = load_document(document_source)
    print(f'Document length (characters): {len(document_text)}')

    print('Chunking document...')
    chunk_records = chunk_text_with_metadata(
        document_text,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        document_title=document_title,
        source=document_source,
        document_id=document_id
    )
    print(f'Number of chunks: {len(chunk_records)}')

    print('Loading embedding model and encoding chunks...')
    embedding_model = load_embedding_model(model_name)
    chunk_texts = [chunk['text'] for chunk in chunk_records]
    embeddings = encode_chunks(embedding_model, chunk_texts)
    print(f'Embeddings shape: {embeddings.shape}')

    print('Building FAISS index...')
    index = build_index(embeddings)
    print(f'FAISS index built. Vectors stored: {index.ntotal}')

    return index, chunk_records, embedding_model
