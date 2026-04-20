from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class RetrievalMethod(str, Enum):
   # Maps directly to the three modules.
    vector  = "vector"    
    keyword = "keyword"   
    clara = "clara"     
    none = "none"  # default for now until it's connected

class UserRegisterRequest(BaseModel):
    # here is what the frontend sends to POST /auth/register
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserLoginRequest(BaseModel):
    # here is what the frontend sends to POST /auth/login
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    #Safe user object never includes password_hash.
    #Returned after register and embedded in TokenResponse.
    id: UUID
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True   # allows building from SQLAlchemy model

class TokenResponse(BaseModel):
    # what the backend returns after a successful login.
    # Patricia's frontend stores the access_token and sends it
    # in the Authorization header for every protected request.
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class DocumentResponse(BaseModel):
    # returned after a document is uploaded or when listing documents.
    id: UUID
    document_id: str
    filename: str
    file_type: Optional[str]
    status: DocumentStatus
    upload_date: datetime
    uploaded_by: UUID

    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    # list of documents belonging to the current user.
    documents: List[DocumentResponse]
    total: int

class ChunkMetadata(BaseModel):
    # Nested metadata inside a chunk that matches shared contract.
    file_type: Optional[str] = None
    uploaded_at: Optional[datetime] = None

class ChunkSchema(BaseModel):
    # Exact mirror of the team's shared chunk schema
    chunk_id: str
    document_id: str
    source_name: str
    text: str
    section: Optional[str] = None
    page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Optional[ChunkMetadata] = None

    class Config:
        from_attributes = True

class RetrievalFilters(BaseModel):
    # Optional filters inside a retrieval request.
    document_ids: Optional[List[str]] = None

class RetrievalOptions(BaseModel):
    # Optional config flags for retrieval.# 
    use_reranking: bool = False

class RetrievalRequest(BaseModel):
    """
    Mirrors team's retrieval_request.schema.json.
    Sent from backend to whichever retrieval adapter is active.
    {
        "query":  "What are the main risks of RAG hallucination?",
        "top_k":  5,
        "filters": { "document_ids": ["doc1", "doc2"] },
        "options": { "use_reranking": false }
    }
    """
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[RetrievalFilters] = None
    options: Optional[RetrievalOptions] = None

class RetrievalResult(BaseModel):
    """
    One result item inside a retrieval response.
    Matches the team's shared retrieval_response schema result item.
    score meaning differs per module:
      vector : cosine similarity  
      keyword: BM25 score         
      clara  : method-specific    
    """
    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float
    rank: int
    metadata: Optional[dict] = None

class RetrievalResponse(BaseModel):
    """
    Mirrors team's retrieval_response.schema.json.
    What each retrieval adapter returns to rag_service.py.
    {
        "query":      "What are the main risks...",
        "method":     "vector",
        "results":    [ ... ],
        "latency_ms": 148
    }
    """
    query: str
    method: RetrievalMethod
    results: List[RetrievalResult]
    latency_ms: Optional[float] = None

class ChatSessionCreate(BaseModel):
    # What frontend sends to create a new session. Title is optional.# 
    title: Optional[str] = "New Chat"

class ChatSessionResponse(BaseModel):
    """
    Returned when a session is created or listed.
    {
        "id": "uuid...",
        "user_id": "uuid...",
        "title": "My first chat",
        "created_at": "2026-03-10T09:00:00Z",
        "updated_at": "2026-03-10T09:05:00Z"
    }
    """
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    """
    One message row returned to the frontend.
    Includes user_id so the frontend always knows who sent it.
    {
        "id": "uuid...",
        "session_id": "uuid...",
        "user_id": "uuid...",
        "role": "user",
        "content": "What is RAG?",
        "created_at": "2026-03-10T09:00:00Z",
        "retrieval_method": null,
        "source_chunk_ids": null
    }
    """
    id: UUID
    session_id: UUID
    user_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    retrieval_method: Optional[RetrievalMethod] = None
    source_chunk_ids: Optional[str] = None   # comma-separated chunk_ids

    class Config:
        from_attributes = True

class ChatHistoryResponse(BaseModel):
    """
    Full conversation returned when frontend loads a session.
    {
        "session": { ...session info... },
        "messages": [ ...all messages... ],
        "total": 12
    }
    """
    session:  ChatSessionResponse
    messages: List[ChatMessageResponse]
    total: int

class QueryRequest(BaseModel):
    """
    What the frontend sends to POST /query.
    Minimum required — just a question:
    {
        "question": "What is RAG?",
        "session_id": "uuid..."        ← optional, auto-created if missing
        "retrieval_method": "vector"   ← optional, defaults to "none" until connected
        "top_k": 5                     ← optional
        "document_ids": ["doc1"]       ← optional, filter to specific docs
    }
    """
    question: str = Field(..., min_length=1)
    session_id: Optional[UUID]  = None
    retrieval_method: Optional[RetrievalMethod] = RetrievalMethod.none
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: Optional[List[str]] = None

class CitationSchema(BaseModel):
    """
    One citation in the final answer.
    Mirrors team's answer_response schema citation object.
    {
        "chunk_id": "doc1_chunk_0004",
        "source_name": "report.pdf",
        "page": 7,
        "section": "Risk Analysis"
    }
    """
    chunk_id: str
    source_name: str
    page: Optional[int] = None
    section: Optional[str] = None

class QueryResponse(BaseModel):
    """
    What the backend returns to the frontend after a query.
    Mirrors team's answer_response schema exactly.
    {
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "citations": [
            {
                "chunk_id": "doc1_chunk_0004",
                "source_name": "report.pdf",
                "page": 7,
                "section": "Risk Analysis"
            }
        ],
        "retrieval_method": "vector",
        "latency_ms": 1130,
        "session_id": "uuid...",
        "user": {
            "id": "uuid...",
            "username": "khalid"
        }
    }
    """
    answer: str
    citations: List[CitationSchema]
    retrieval_method: RetrievalMethod
    latency_ms: Optional[float] = None
    session_id: UUID
    user: UserResponse

class LogResponse(BaseModel):
    # returned when querying the activity log.
    id: UUID
    user_id: Optional[UUID]
    action: str
    detail: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True