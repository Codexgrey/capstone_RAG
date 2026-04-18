# rag_service.py — runs the full RAG query flow
# steps:
#   1. search ChromaDB for chunks
#   2. fetch doc info from Postgres
#   3. build prompt from chunks
#   4. call LLM for answer
#   5. format response + citations
#   6. save Q&A to chat_messages
#   7. return final response
# used by: app/api/query.py
import time
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from app.models.db_models import ChatSession, ChatMessage, Document, Log
from app.generation.prompt_builder import build_chat_prompt
from app.generation.llm_client import generate_answer
from app.generation.response_formatter import format_response, format_error_response


def handle_query(
    db: Session,
    user_id,
    question: str,
    session_id: Optional[str] = None,
    retrieval_method: str = "vector",
    top_k: int = 5,
    document_ids: list = None,
) -> dict:
    """
    Full RAG query pipeline.

    Args:
        db : PostgreSQL session
        user_id : UUID of the logged-in user
        question : The user's question
        session_id : Chat session UUID (created if None)
        retrieval_method : "vector" | "keyword" | "clara" | "none"
        top_k : Number of chunks to retrieve
        document_ids : Optional filter to specific documents

    Returns:
        Formatted response dict with answer, citations, latency_ms
    """
    start_time = time.time()

    # 1: Get or create chat session 
    session = _get_or_create_session(db, user_id, session_id)

    # 2: Save user message to PostgreSQL 
    _save_message(
        db         = db,
        session_id = session.id,
        user_id    = user_id,
        role       = "user",
        content    = question
    )

    # 3: Retrieve relevant chunks
    chunks = []
    actual_method = "none"

    if retrieval_method == "vector":
        chunks, actual_method = _retrieve_vector(question, top_k, document_ids)
    elif retrieval_method == "keyword":
        # Olivier's keyword retrieval, not connected yet
        chunks, actual_method = _retrieve_keyword(question, top_k)
    elif retrieval_method == "clara":
        # Nathan's CLaRA retrieval, not connected yet
        chunks, actual_method = _retrieve_clara(question, top_k)

    # 4: Enrich chunks with PostgreSQL source info 
    chunks = _enrich_chunks_with_source(db, chunks)

    # 5: Build prompt and call LLM 
    messages = build_chat_prompt(question=question, chunks=chunks)

    try:
        answer = generate_answer(prompt=question, messages=messages)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # 6: Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # 7: Save assistant response to PostgreSQL 
    source_chunk_ids = ",".join([c.get("chunk_id", "") for c in chunks])

    _save_message(
        db = db,
        session_id = session.id,
        user_id = user_id,
        role = "assistant",
        content = answer,
        retrieval_method = actual_method,
        source_chunk_ids = source_chunk_ids
    )

    # 8: Log the query 
    db.add(Log(
        user_id = user_id,
        action = "query_sent",
        detail = (
            f"question={question[:50]} "
            f"method={actual_method} "
            f"chunks={len(chunks)} "
            f"latency={round(latency_ms)}ms"
        ),
        timestamp = datetime.utcnow()
    ))
    db.commit()

    # 9: Format and return response
    return format_response(
        answer = answer,
        chunks = chunks,
        retrieval_method = actual_method,
        latency_ms = latency_ms,
        session_id = str(session.id)
    )


# Retrieval methods 
def _retrieve_vector(question: str, top_k: int, document_ids: list = None):
    # Vector semantic search using ChromaDB.
    # Returns chunks ranked by cosine similarity.
    try:
        from app.ingestion.indexer import search_chunks
        chunks = search_chunks(
            query = question,
            top_k = top_k,
            document_ids = document_ids
        )
        return chunks, "vector"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️  Vector retrieval failed: {e} — falling back to PostgreSQL")
        return _fallback_postgres_chunks(top_k), "none"


def _retrieve_keyword(question: str, top_k: int):
    print("  🔒 Keyword retrieval not connected yet — using fallback")
    return [], "keyword"

def _retrieve_clara(question: str, top_k: int):
    print("  🔒 CLaRA retrieval not connected yet — using fallback")
    return [], "clara"

def _fallback_postgres_chunks(top_k: int = 5):
    # Fallback when ChromaDB is not available.
    # Returns raw chunks from PostgreSQL without semantic ranking.
    from app.config.database import SessionLocal
    from app.models.db_models import DocumentChunk

    db = SessionLocal()
    try:
        raw_chunks = db.query(DocumentChunk).limit(top_k).all()
        return [
            {
                "chunk_id" : c.chunk_id,
                "document_id" : c.document_id,
                "source_name" : c.source_name,
                "text" : c.text,
                "score": 1.0,
                "rank" : i + 1,
                "metadata" : {
                    "page" : c.page,
                    "document_id" : c.document_id
                }
            }
            for i, c in enumerate(raw_chunks)
        ]
    finally:
        db.close()

# Helpers
def _enrich_chunks_with_source(db: Session, chunks: list) -> list:
    # Add filename and upload info from PostgreSQL to each chunk.
    # Uses document_id from chunk metadata to look up the documents table.
    # This is the PostgreSQL ↔ ChromaDB bridge in action.
    if not chunks:
        return chunks

    enriched = []
    for chunk in chunks:
        doc_id = chunk.get("document_id") or \
                 chunk.get("metadata", {}).get("document_id", "")

        if doc_id:
            doc = db.query(Document).filter(Document.document_id == doc_id).first()

            if doc:
                chunk["filename"]    = doc.filename
                chunk["uploaded_by"] = str(doc.uploaded_by)
                chunk["upload_date"] = str(doc.upload_date)

        enriched.append(chunk)

    return enriched

def _get_or_create_session(db: Session, user_id, session_id: str = None):
    # Get existing session or create a new one.
    # If session_id is None, creates a new session automatically.
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id      == session_id,
            ChatSession.user_id == user_id
        ).first()
        if session:
            return session

    # Create new session
    new_session = ChatSession(
        id = uuid.uuid4(),
        user_id = user_id,
        title = "New Chat",
        created_at = datetime.utcnow(),
        updated_at = datetime.utcnow()
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def _save_message(
    db,
    session_id,
    user_id,
    role: str,
    content: str,
    retrieval_method: str = None,
    source_chunk_ids: str = None
):
    # Save a single message to the chat_messages table.# 
    db.add(ChatMessage(
        id = uuid.uuid4(),
        session_id = session_id,
        user_id = user_id,
        role = role,
        content = content,
        created_at = datetime.utcnow(),
        retrieval_method = retrieval_method,
        source_chunk_ids = source_chunk_ids
    ))
    db.commit()