# import chromadb
# 
# # creates a client that saves data to disk
# client = chromadb.PersistentClient(path="./chroma_storage")
# # enusres i have a collection named documents
# collection = client.get_or_create_collection(name="documents")
# 
# # using Chromadb own embedding for learning
# # add a chunk
# collection.add(
#     ids=["chunk_001"],
#     documents=["Rag stands for Retrieval Augmented Generation."],
#     metadatas=[{"document_id": "doc_abc", "source_name": "test.txt", "page": 1}]
# )
# 
# print("collection name: ", collection.name)
# # how many chunks are stored
# print("Chunks stored: ", collection.count())
# 
# # search by meaning
# results = collection.query(
#     query_texts=["What does RAG mean?"],
#     n_results=1
# )
# 
# print("Result: ", results["documents"])
# print("Metadata: ", results["metadatas"])
# 
from typing import List, Dict, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.config.settings import settings

_chroma_client = None
_collection = None
_embedding_model = None
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-miniLM-L6-v2"

# creates a client that saves data to disk
def get_chroma_client():
    
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path = settings.CHROMA_PERSIST_DIR
        )
        print(f"  ✅ ChromaDB client initialized → {settings.CHROMA_PERSIST_DIR}")
    return _chroma_client

# enusres i have a collection named documents
def get_collection():

    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name = COLLECTION_NAME,
            # use cosine similarity
            metadata = {"hnsw:space": "cosine"}
        )
        print(f"  ✅ ChromaDB collection ready → '{COLLECTION_NAME}'")
    return _collection

def get_embedding_model():
    # Returns the sentence-transformers embedding model.
    # Downloads on first use, cached locally after that.
    global _embedding_model
    if _embedding_model is None:
        print(f"  ⏳ Loading embedding model '{EMBEDDING_MODEL}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  ✅ Embedding model loaded")
    return _embedding_model

def index_chunks(
    chunks: List[Dict[str, Any]],
    document_id: str,
    uploaded_by: str = "",
    file_type: str = "",
) -> Dict[str, Any]:

    if not chunks:
        raise ValueError("No chunks to index")

    collection = get_collection()
    model = get_embedding_model()

    # extract texts
    texts = [chunk["text"] for chunk in chunks]

    print(f"⏳ Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    print(f"✅ Embeddings created")

    # build the chromadb inputs
    ids = []
    documents = []
    metadatas = []

    uploaded_at = datetime.utcnow().isoformat() + "Z"

    for chunk, embedding in zip(chunks, embeddings):
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "chunk_id" : chunk["chunk_id"],
            "document_id" : document_id,          # ← bridge to PostgreSQL
            "source_name" : chunk["source_name"],
            "page" : chunk.get("page", 1),
            "start_char" : chunk.get("start_char", 0),
            "end_char" : chunk.get("end_char", 0),
            "file_type" : file_type,
            "uploaded_by" : uploaded_by,
            "uploaded_at" : uploaded_at,
        })    

    # stonre in chromadb
    collection.upsert(
        ids = ids,
        documents = documents,
        embeddings = embeddings,
        metadatas = metadatas,
    )

    print(f"✅{len(chunks)} chunks stored in ChromaDB")
 
    return {
        "chunks_stored": len(chunks),
        "collection": COLLECTION_NAME,
        "document_id": document_id,
        "metadata": {
            "file_type": file_type,
            "uploaded_by": uploaded_by,
            "uploaded_at": uploaded_at,
        }
    }

def delete_document_chunks(document_id: str) -> int:
    # Delete all ChromaDB chunks for a document.
    # Called when a document is deleted from PostgreSQL.

    collection = get_collection()
 
    # Find all chunks for this document
    results = collection.get(
        where={"document_id": document_id}
    )
 
    if not results["ids"]:
        return 0
 
    collection.delete(ids=results["ids"])
    print(f"🗑️  Deleted {len(results['ids'])} chunks for document {document_id}")
    return len(results["ids"])
 
# Added the search chunk function
def search_chunks(
    query: str,
    top_k: int = 5,
    document_ids: list = None,
) -> List[Dict[str, Any]]:
    # Search ChromaDB for chunks similar to the query.
    collection = get_collection()
    model = get_embedding_model()
 
    # Embed the query using same model as documents
    query_embedding = model.encode([query]).tolist()
 
    # Build optional filter
    where_filter = None
    if document_ids:
        where_filter = {"document_id": {"$in": document_ids}}
 
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
 
    # Format results to match team's shared retrieval_response contract
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
            # Convert cosine distance to similarity score (1 = identical)
            score = round(1 - distance, 4)
            formatted.append({
                "chunk_id" : chunk_id,
                "document_id" : metadata.get("document_id", ""),
                "source_name" : metadata.get("source_name", ""),
                "text" : text,
                "score" : score,
                "rank" : rank,
                "metadata" : metadata,
            })
 
    return formatted
 