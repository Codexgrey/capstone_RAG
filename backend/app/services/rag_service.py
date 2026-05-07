"""
services/rag_service.py — RAG Query Orchestration

Routes queries to the correct retrieval module (vector, keyword, or hybrid),
builds the LLM prompt, generates the answer, and persists chat history.

Special intent: queries asking to list ingested documents are intercepted
before retrieval and answered directly from the PostgreSQL document registry.
"""

import re
import time
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from app.models.db_models import ChatSession, ChatMessage, Document, Log
from app.generation.prompt_builder import build_chat_prompt
from app.generation.llm_client import generate_answer
from app.generation.response_formatter import format_response


# ── Document-list intent keywords ─────────────────────────────────────────────
_LIST_PATTERNS = re.compile(
    r"\b(list|show|what|which|how many|tell me).{0,30}"
    r"(document|doc|file|paper|context|ingested|uploaded|available|in memory|you have)\b",
    re.IGNORECASE,
)


def _is_document_list_query(question: str) -> bool:
    """Return True when the user is asking what documents are currently ingested."""
    return bool(_LIST_PATTERNS.search(question))


def _build_document_list_answer(db: Session, user_id) -> str:
    """Build a human-readable list of all completed documents for this user."""
    docs = (
        db.query(Document)
        .filter(Document.uploaded_by == user_id, Document.status == "completed")
        .order_by(Document.upload_date.asc())
        .all()
    )
    if not docs:
        return (
            "There are currently **no documents** ingested in the system. "
            "Please upload a document using the sidebar to get started."
        )

    lines = [f"I currently have **{len(docs)} document(s)** in context:\n"]
    for i, d in enumerate(docs, start=1):
        dt = d.upload_date.strftime("%Y-%m-%d %H:%M UTC") if d.upload_date else "unknown"
        lines.append(f"{i}. **{d.filename}** ({d.file_type.upper()}) — ingested {dt}")

    return "\n".join(lines)


def handle_query(
    db: Session,
    user_id,
    question: str,
    session_id: Optional[str] = None,
    retrieval_method: str = "vector",
    top_k: int = 3,
    document_ids: list = None,
) -> dict:

    start_time = time.time()

    # 1 — Get or create session
    session = _get_or_create_session(db, user_id, session_id, first_question=question)

    # 2 — Save user message
    _save_message(db=db, session_id=session.id, user_id=user_id,
                  role="user", content=question)

    # 3 — Short-circuit: document list intent
    if _is_document_list_query(question):
        answer = _build_document_list_answer(db, user_id)
        latency_ms = (time.time() - start_time) * 1000
        _save_message(db=db, session_id=session.id, user_id=user_id,
                      role="assistant", content=answer,
                      retrieval_method=None, source_chunk_ids="")
        db.add(Log(
            user_id=user_id, action="query_sent",
            detail=f"question={question[:50]} method=system(doc_list) latency={round(latency_ms)}ms",
            timestamp=datetime.utcnow()
        ))
        db.commit()
        return format_response(
            answer=answer, chunks=[], retrieval_method=None,
            latency_ms=latency_ms, session_id=str(session.id), question=question,
        )

    # 4 — Retrieve chunks via the requested method
    chunks, actual_method = _dispatch_retrieval(
        retrieval_method, question, top_k, document_ids
    )

    # 5 — Enrich with PostgreSQL source info
    chunks = _enrich_chunks_with_source(db, chunks)

    # 6 — Build prompt and call LLM
    messages = build_chat_prompt(question=question, chunks=chunks)
    try:
        answer = generate_answer(prompt=question, messages=messages, temperature=0.1)
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # 7 — Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # 8 — Save assistant message
    source_chunk_ids = ",".join([c.get("chunk_id", "") for c in chunks])
    _save_message(db=db, session_id=session.id, user_id=user_id,
                  role="assistant", content=answer,
                  retrieval_method=actual_method,
                  source_chunk_ids=source_chunk_ids)

    # 9 — Log
    db.add(Log(
        user_id=user_id, action="query_sent",
        detail=(f"question={question[:50]} method={actual_method} "
                f"chunks={len(chunks)} latency={round(latency_ms)}ms"),
        timestamp=datetime.utcnow()
    ))
    db.commit()

    # 10 — Return formatted response
    return format_response(
        answer=answer,
        chunks=chunks,
        retrieval_method=actual_method,
        latency_ms=latency_ms,
        session_id=str(session.id),
        question=question,
    )


# ── Retrieval dispatcher ──────────────────────────────────────────────────────

def _dispatch_retrieval(method: str, question: str, top_k: int, document_ids: list):
    """Route to the correct retrieval module. Falls back to ChromaDB if all fail."""
    if method == "vector":
        return _retrieve_vector(question, top_k, document_ids)
    elif method == "keyword":
        return _retrieve_keyword(question, top_k, document_ids)
    elif method == "hybrid":
        return _retrieve_hybrid(question, top_k, document_ids)
    else:
        return _retrieve_chroma_fallback(question, top_k, document_ids), "none"


def _retrieve_vector(question: str, top_k: int, document_ids: list = None):
    """Collins's FAISS vector retrieval, with ChromaDB fallback."""
    try:
        from app.retrieval.vector_adapter import retrieve as collins_retrieve
        chunks = collins_retrieve(query=question, top_k=top_k)
        if chunks:
            print(f"  ✅ Vector (FAISS) — {len(chunks)} chunks")
            return chunks, "vector"
        print("  ⚠️  Vector FAISS returned no results — trying ChromaDB")
    except FileNotFoundError:
        print("  ⚠️  No FAISS index yet — trying ChromaDB")
    except Exception as e:
        print(f"  ⚠️  Vector retrieval error: {e} — trying ChromaDB")

    chunks = _retrieve_chroma_fallback(question, top_k, document_ids)
    if chunks:
        print(f"  ✅ Vector (ChromaDB) — {len(chunks)} chunks")
    return chunks, "vector"


def _retrieve_keyword(question: str, top_k: int, document_ids: list = None):
    """Olivier's BM25 keyword retrieval, with ChromaDB fallback."""
    try:
        from app.retrieval.keyword_adapter import retrieve as olivier_retrieve
        chunks = olivier_retrieve(query=question, top_k=top_k)
        if chunks:
            print(f"  ✅ Keyword (BM25) — {len(chunks)} chunks")
            return chunks, "keyword"
        print("  ⚠️  Keyword BM25 returned no results — falling back to ChromaDB")
    except FileNotFoundError:
        print("  ⚠️  No keyword index yet — falling back to ChromaDB")
    except Exception as e:
        print(f"  ⚠️  Keyword retrieval error: {e} — falling back to ChromaDB")

    # ChromaDB is always current (updated on every upload) — use it as fallback
    chunks = _retrieve_chroma_fallback(question, top_k, document_ids)
    if chunks:
        print(f"  ✅ Keyword fallback (ChromaDB) — {len(chunks)} chunks")
    return chunks, "keyword"


def _retrieve_hybrid(question: str, top_k: int, document_ids: list = None):
    """Nathan's FAISS+BM25+RRF hybrid retrieval, with ChromaDB fallback."""
    try:
        from app.retrieval.hybrid_adapter import retrieve as nathan_retrieve
        chunks = nathan_retrieve(query=question, top_k=top_k)
        if chunks:
            print(f"  ✅ Hybrid (FAISS+BM25+RRF) — {len(chunks)} chunks")
            return chunks, "hybrid"
        print("  ⚠️  Hybrid returned no results — falling back to ChromaDB")
    except FileNotFoundError:
        print("  ⚠️  No hybrid index yet — falling back to ChromaDB")
    except Exception as e:
        print(f"  ⚠️  Hybrid retrieval error: {e} — falling back to ChromaDB")

    chunks = _retrieve_chroma_fallback(question, top_k, document_ids)
    if chunks:
        print(f"  ✅ Hybrid fallback (ChromaDB) — {len(chunks)} chunks")
    return chunks, "hybrid"


def _retrieve_chroma_fallback(question: str, top_k: int = 3, document_ids: list = None):
    """
    ChromaDB semantic search — always current because every upload indexes here.
    This is the universal fallback when FAISS/BM25 indexes are not ready yet.
    """
    try:
        from app.ingestion.indexer import search_chunks
        chunks = search_chunks(query=question, top_k=top_k, document_ids=document_ids)
        return chunks
    except Exception as e:
        print(f"  ⚠️  ChromaDB fallback failed: {e}")
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _enrich_chunks_with_source(db: Session, chunks: list) -> list:
    if not chunks:
        return chunks
    enriched = []
    for chunk in chunks:
        doc_id = chunk.get("document_id") or chunk.get("metadata", {}).get("document_id", "")
        if doc_id:
            doc = db.query(Document).filter(Document.document_id == doc_id).first()
            if doc:
                chunk["filename"]    = doc.filename
                chunk["uploaded_by"] = str(doc.uploaded_by)
                chunk["upload_date"] = str(doc.upload_date)
        enriched.append(chunk)
    return enriched


def _get_or_create_session(db: Session, user_id, session_id: str = None, first_question: str = ""):
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id
        ).first()
        if session:
            return session

    title = (first_question[:40] + "...") if len(first_question) > 40 else first_question
    title = title or "New Chat"

    new_session = ChatSession(
        id         = uuid.uuid4(),
        user_id    = user_id,
        title      = title,
        created_at = datetime.utcnow(),
        updated_at = datetime.utcnow()
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def _save_message(db, session_id, user_id, role: str, content: str,
                  retrieval_method: str = None, source_chunk_ids: str = None):
    db.add(ChatMessage(
        id=uuid.uuid4(), session_id=session_id, user_id=user_id,
        role=role, content=content, created_at=datetime.utcnow(),
        retrieval_method=retrieval_method, source_chunk_ids=source_chunk_ids
    ))
    db.commit()
