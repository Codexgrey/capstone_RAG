"""
src/retrieval/retriever.py
Similarity search logic.
Embeds a query and retrieves the top-k most similar chunks from the FAISS index.

Pipeline position:
    query → [retriever] → top-k chunks → return results
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


def l2_to_similarity(distance: float) -> float:
    """
    Convert an L2 distance to a similarity score in the range (0, 1].

    Args:
        distance: L2 distance value (non-negative).

    Returns:
        Similarity score — 1.0 is identical, approaches 0 for large distances.
    """
    return 1 / (1 + distance)


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunk_records: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top-k most similar chunks from the FAISS index.

    Args:
        query:         Natural language query string.
        model:         Loaded SentenceTransformer embedding model.
        index:         Populated FAISS index.
        chunk_records: List of chunk metadata dicts produced by the chunker.
        top_k:         Number of top results to return.

    Returns:
        List of result dicts ordered by similarity (highest first), each containing:
            rank, document_id, document_title, source, chunk_id, chunk_index,
            word_count, distance, similarity, citation, text.

    Raises:
        ValueError: If the query is empty or whitespace.
    """
    if not query or not query.strip():
        raise ValueError('Query cannot be empty.')

    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype='float32')

    safe_top_k = min(top_k, len(chunk_records))
    distances, indices = index.search(query_vector, safe_top_k)

    results = []
    for rank, chunk_idx in enumerate(indices[0]):
        chunk = chunk_records[int(chunk_idx)]
        distance_value = float(distances[0][rank])
        similarity_value = float(l2_to_similarity(distance_value))

        results.append({
            'rank': rank + 1,
            'document_id': chunk['document_id'],
            'document_title': chunk['document_title'],
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id'],
            'chunk_index': chunk['chunk_index'],
            'word_count': chunk['word_count'],
            'distance': distance_value,
            'similarity': similarity_value,
            'citation': f"[{chunk['document_title']} | {chunk['chunk_id']}]",
            'text': chunk['text']
        })

    return results
