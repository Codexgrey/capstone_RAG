import numpy as np

def l2_to_similarity(distance):
    """
    Converts L2 distance to a similarity score between 0 and 1.
    Lower distance = higher similarity.
    """
    return 1 / (1 + distance)


def retrieve(query, model, index, chunk_records, top_k=3):
    """
    Embeds the query, searches the FAISS index,
    and returns the top-k most similar chunks.
    """
    if not query or not query.strip():
        raise ValueError('Query cannot be empty.')

    # Embed the query using the same model used for chunks
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
            'text': chunk['text'],
            'metadata': chunk['metadata']
        })

    return results