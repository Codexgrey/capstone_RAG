"""
src/indexing/vector_store.py
FAISS vector database setup and indexing logic.
Builds and manages a flat L2 index for similarity search.
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
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
