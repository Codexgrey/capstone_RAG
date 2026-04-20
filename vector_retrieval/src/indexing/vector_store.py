"""
src/indexing/vector_store.py
FAISS vector database setup, indexing logic, and persistence.
Builds, saves, and loads a flat L2 index for similarity search.
"""

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS IndexFlatL2 index from a set of embeddings.

    Args:
        embeddings: Float32 NumPy array of shape (num_chunks, embedding_dim).

    Returns:
        FAISS index with all embeddings added.

    Raises:
        ValueError: If the embeddings array is empty or None.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError('Embeddings array is empty. Cannot build FAISS index.')

    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def build_and_save_index(
    embeddings: np.ndarray,
    chunk_records: list,
    index_path: str,
    chunks_path: str
) -> faiss.IndexFlatL2:
    """
    Build a FAISS index, then persist it and the chunk records to disk.

    Args:
        embeddings:    Float32 embeddings array.
        chunk_records: List of chunk metadata dicts from the chunker.
        index_path:    File path to save the FAISS index (e.g. 'faiss_index.bin').
        chunks_path:   File path to save chunk records (e.g. 'chunk_records.npy').

    Returns:
        The built FAISS index.
    """
    index = build_index(embeddings)
    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(chunk_records, dtype=object))

    print(
        f'  FAISS index saved to   : {index_path}\n'
        f'  Chunk records saved to : {chunks_path}\n'
        f'  Total vectors in index : {index.ntotal}\n'
    )
    return index


def load_index(index_path: str, chunks_path: str) -> tuple:
    """
    Reload a previously saved FAISS index and chunk records from disk.

    Use this to skip re-embedding when documents have not changed.

    Args:
        index_path:  Path to the saved FAISS index file.
        chunks_path: Path to the saved chunk records .npy file.

    Returns:
        Tuple of (faiss_index, chunk_records list).
    """
    index         = faiss.read_index(index_path)
    chunk_records = np.load(chunks_path, allow_pickle=True).tolist()

    print(
        f'  Index loaded from  : {index_path}  ({index.ntotal} vectors)\n'
        f'  Chunks loaded from : {chunks_path}  ({len(chunk_records)} records)\n'
    )
    return index, chunk_records
