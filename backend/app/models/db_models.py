"""
db_models.py: all PostgreSQL tables.
Uses SQLAlchemy Base from config.database.

Includes:
- User: registered accounts
- Document: uploaded file metadata
- DocumentChunk: chunk info (matches shared contract)
- ChatSession: groups messages into conversations
- ChatMessage: stores prompts + answers with user_id
- Log: records system activity
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Text,
    DateTime, ForeignKey, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.config.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String(50),  unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    documents = relationship("Document", back_populates="owner",   cascade="all, delete")
    chat_sessions = relationship("ChatSession", back_populates="user",    cascade="all, delete")
    logs = relationship("Log", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


class Document(Base):
    """
    document_id is the shared UUID that links PostgreSQL to ChromaDB.
    Every chunk in ChromaDB stores this ID in its metadata.
    """
    __tablename__ = "documents"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(String(100), unique=True, nullable=False, index=True,
                         comment="Shared key: PostgreSQL ↔ ChromaDB ↔ keyword ↔ CLaRA")
    filename    = Column(String(255), nullable=False)
    filepath    = Column(String(500), nullable=False)
    file_type   = Column(String(20),  nullable=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    status      = Column(
        SAEnum("uploaded", "processing", "completed", "failed", name="doc_status"),
        default="uploaded", nullable=False
    )

    owner  = relationship("User",          back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete")

    def __repr__(self):
        return f"<Document document_id={self.document_id} filename={self.filename}>"


class DocumentChunk(Base):
    # Mirrors the team's shared chunk.schema.json.
    # chunk_id is stored here AND in ChromaDB metadata — links citations back to source.
    
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String(150), unique=True, nullable=False, index=True)
    document_id = Column(String(100), ForeignKey("documents.document_id", ondelete="CASCADE"),
                         nullable=False, index=True)
    source_name = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String(255), nullable=True)
    page = Column(Integer, nullable=True)
    start_char  = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk chunk_id={self.chunk_id} page={self.page}>"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    title = Column(String(255), default="New Chat", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user     = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete")

    def __repr__(self):
        return f"<ChatSession id={self.id} user_id={self.user_id}>"


class ChatMessage(Base):
    """
    Every user prompt and assistant answer is one row here.

    WHO asked : user_id (FK: users.id)
    WHAT asked: content (role='user')
    ANSWER : content (role='assistant')
    HOW answered : retrieval_method (vector | keyword | clara — filled in Step 9)
    WHAT CITED: source_chunk_ids (comma-separated chunk_ids)
    """
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(SAEnum("user", "assistant", "system", name="message_role"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Filled in once retrieval modules connect, for now i just put the default to none
    retrieval_method = Column(
        SAEnum("vector", "keyword", "clara", "none", name="retrieval_method_enum"),
        nullable=True, default=None
    )
    source_chunk_ids = Column(Text, nullable=True,
                              comment="Comma-separated chunk_ids used to generate this answer")

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage id={self.id} role={self.role} user_id={self.user_id}>"


class Log(Base):
    __tablename__ = "logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"),
                       nullable=True, index=True)
    action = Column(String(100), nullable=False)
    detail = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="logs")

    def __repr__(self):
        return f"<Log id={self.id} action={self.action} user_id={self.user_id}>"