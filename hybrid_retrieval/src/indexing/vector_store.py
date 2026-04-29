import faiss
import numpy as np

def build_and_save_index(embeddings, chunk_records, index_path, chunks_path):
    """
    Builds a FAISS index from embeddings and saves it to disk.
    Also saves chunk records alongside the index.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError('Embeddings array is empty. Cannot build FAISS index.')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(chunks_path, np.array(chunk_records, dtype=object))

    print(f'FAISS index saved to   : {index_path}')
    print(f'Chunk records saved to : {chunks_path}')
    print(f'Total vectors in index : {index.ntotal}')

    return index


def load_index(index_path, chunks_path):
    """
    Reloads a previously saved FAISS index from disk.
    Use this to skip re-embedding when documents haven't changed.
    """
    index = faiss.read_index(index_path)
    chunk_records = np.load(chunks_path, allow_pickle=True).tolist()

    print(f'Index loaded from  : {index_path} ({index.ntotal} vectors)')
    print(f'Chunks loaded from : {chunks_path} ({len(chunk_records)} records)')

    return index, chunk_records