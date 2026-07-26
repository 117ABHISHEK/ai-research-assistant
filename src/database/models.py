import uuid
import datetime

from sqlalchemy import Column, String, Integer, DateTime, Text
from src.database.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=generate_uuid)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")  # PENDING, PROCESSED, FAILED
    category = Column(String, nullable=True)  # filled in by TF classifier


class QueryLog(Base):
    """Tracks every question asked, for analytics (most-queried docs, total questions)."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    doc_ids_referenced = Column(Text, nullable=True)  # comma-separated doc_ids from citations
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class ConversationMemory(Base):
    """Optional persistence layer for session history (in-memory dict is primary store at runtime)."""
    __tablename__ = "conversation_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)