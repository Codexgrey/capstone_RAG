# query.py API routes for questions + chat history
# endpoints:
#   POST /query: ask question, get answer
#   POST /chat/sessions: create new chat session
#   GET  /chat/sessions: list user’s sessions
#   GET  /chat/sessions/{id}: get messages in a session
# frontend gets from POST /query:
#   { "answer": "...", "citations": [...], "retrieval_method": "...",
#     "latency_ms": 320.5, "session_id": "uuid..." }
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.config.dependencies import get_current_user
from app.models.db_models import User, ChatSession, ChatMessage
from app.models.schemas import QueryRequest
from app.services.rag_service import handle_query

router = APIRouter(tags=["Query & Chat"])


# POST /query 
@router.post("/query")
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # main RAG query endpoint
    # needs: Authorization Bearer <token>
    # body fields:
    #   "question": user’s question
    #   "session_id": optional UUID (auto‑created if missing)
    #   "retrieval_method": optional (default "vector")
    #   "top_k": optional (default 5)

    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    result = handle_query(
        db               = db,
        user_id          = current_user.id,
        question         = payload.question,
        session_id       = str(payload.session_id) if payload.session_id else None,
        retrieval_method = payload.retrieval_method.value
                           if hasattr(payload.retrieval_method, 'value')
                           else str(payload.retrieval_method or "vector"),
        top_k = payload.top_k or 5,
        document_ids = payload.document_ids,
    )

    return result


# POST /chat/sessions 
@router.post("/chat/sessions", status_code=status.HTTP_201_CREATED)
def create_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Create a new empty chat session.
    session = ChatSession(
        id = uuid.uuid4(),
        user_id = current_user.id,
        title = "New Chat",
        created_at = datetime.utcnow(),
        updated_at = datetime.utcnow()
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return {
        "session_id": str(session.id),
        "title":      session.title,
        "created_at": session.created_at
    }


# GET /chat/sessions 
@router.get("/chat/sessions")
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # List all chat sessions for the current user.
    # Ordered by most recently updated first.
    sessions = db.query(ChatSession)\
                 .filter(ChatSession.user_id == current_user.id)\
                 .order_by(ChatSession.updated_at.desc())\
                 .all()

    return {
        "sessions": [
            {
                "session_id": str(s.id),
                "title":      s.title,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ],
        "total": len(sessions)
    }


# GET /chat/sessions/{session_id}
@router.get("/chat/sessions/{session_id}")
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get all messages in a specific session.
    # Returns messages in chronological order.
    # Patricia's frontend loads to display chat history.
    
    # Verify session belongs to this user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    messages = db.query(ChatMessage)\
                 .filter(ChatMessage.session_id == session_id)\
                 .order_by(ChatMessage.created_at.asc())\
                 .all()

    return {
        "session_id" : str(session.id),
        "title" : session.title,
        "messages": [
            {
                "id" : str(m.id),
                "role" : m.role,
                "content" : m.content,
                "created_at" : m.created_at,
                "retrieval_method" : m.retrieval_method,
                "source_chunk_ids" : m.source_chunk_ids,
            }
            for m in messages
        ],
        "total": len(messages)
    }