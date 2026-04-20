"""
src/indexing/indexer.py
Orchestrates the full multi-document indexing pipeline:
  scan folder → load → chunk → embed → build index → persist.
"""

import os
from src.utils.loader import load_document, get_files_from_folder
from src.utils.chunker import chunk_text_with_metadata
from src.models.embedding_model import load_embedding_model, encode_chunks
from src.indexing.vector_store import build_and_save_index
from typing import Tuple, List, Dict, Any
import faiss
import numpy as np


# Default persistence paths
INDEX_SAVE_PATH  = 'faiss_index.bin'
CHUNKS_SAVE_PATH = 'chunk_records.npy'


def build_pipeline(
    document_paths: List[str],
    chunk_size: int = 150,
    chunk_overlap: int = 30,
    model_name: str = 'all-MiniLM-L6-v2',
    index_path: str = INDEX_SAVE_PATH,
    chunks_path: str = CHUNKS_SAVE_PATH,
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]], object]:
    """
    Run the full indexing pipeline across one or more documents.

    Steps:
        1. Load and parse each file (auto-detects format)
        2. Chunk into overlapping word windows with metadata
        3. Encode all chunks into embedding vectors
        4. Build FAISS index and persist to disk

    Args:
        document_paths: List of file paths to ingest.
        chunk_size:     Number of words per chunk.
        chunk_overlap:  Number of overlapping words between chunks.
        model_name:     SentenceTransformer model identifier.
        index_path:     Path to save the FAISS index.
        chunks_path:    Path to save chunk records.

    Returns:
        Tuple of (faiss_index, all_chunk_records, embedding_model)
    """
    all_chunk_records = []
    ingestion_log     = []

    print('Starting ingestion...\n')
    print('=' * 80)
    
    for doc_index, path in enumerate(document_paths):
        doc_id    = f'doc-{doc_index + 1:03d}'
        doc_title = os.path.splitext(os.path.basename(path))[0]\
                      .replace('_', ' ').replace('-', ' ').title()

        try:
            text, file_metadata = load_document(path)
            chunks = chunk_text_with_metadata(
                text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                document_title=doc_title,
                source=file_metadata['file_name'],
                document_id=doc_id,
                file_metadata=file_metadata,
            )
            all_chunk_records.extend(chunks)

            ingestion_log.append({
                'document_id':    doc_id,
                'document_title': doc_title,
                'file_name':      file_metadata['file_name'],
                'file_type':      file_metadata['file_type'],
                'chunks':         len(chunks),
                'status':         'OK',
            })

            print(
                f"[{doc_id}] {doc_title} \n"
                f"  File       : {file_metadata['file_name']} \n"
                f"  Type       : {file_metadata['file_type']} \n"
                f"  Size       : {file_metadata['file_size_kb']} KB \n"
                f"  Characters : {len(text)} \n"
                f"  Chunks     : {len(chunks)} \n"
                f"  Status     : OK\n"
            )
        
        except Exception as e:
            ingestion_log.append({'document_id': doc_id, 'file_name': path, 'status': f'FAILED: {e}'})
            print(
                f"[{doc_id}] FAILED: {path}"
                f"  Error: {e} \n"
            )

    print('=' * 80)
    print(
        f"Ingestion complete. \n"
        f"Documents processed : {len(document_paths)} \n"
        f"Total chunks        : {len(all_chunk_records)} \n"
    )

    print('Loading embedding model and encoding chunks...')
    embedding_model = load_embedding_model(model_name)
    chunk_texts     = [chunk['text'] for chunk in all_chunk_records]
    embeddings      = encode_chunks(embedding_model, chunk_texts)
    print(f'Embeddings shape : {embeddings.shape}')
    print(f'Embedding dtype  : {embeddings.dtype}\n')

    print('Building and saving FAISS index...')
    index = build_and_save_index(embeddings, all_chunk_records, index_path, chunks_path)
    print()

    return index, all_chunk_records, embedding_model
