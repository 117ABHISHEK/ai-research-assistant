from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import datetime
import logging

from src.database.base import get_db
from src.database.models import QueryLog
from src.vector_store.manager import vector_store_manager
from src.rag.qa_chain import rag_qa

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search & Question Answering"])


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_ids: Optional[List[str]] = None


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = None


@router.post("/semantic")
async def semantic_search(request: SearchRequest):
    """Semantic (vector similarity) search across indexed document chunks."""
    try:
        results = vector_store_manager.semantic_search(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
        return {"query": request.query, "mode": "semantic", "results": results}
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail="Semantic search failed.")


@router.post("/keyword")
async def keyword_search(request: SearchRequest):
    """Keyword (exact substring) search across indexed document chunks.

    Best for: exact terms, names, error codes, or identifiers where semantic
    similarity might miss precise matches. Semantic search is better for
    conceptual/paraphrased queries. Use keyword search when the user knows
    the precise term they're looking for.
    """
    try:
        results = vector_store_manager.keyword_search(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
        return {"query": request.query, "mode": "keyword", "results": results}
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        raise HTTPException(status_code=500, detail="Keyword search failed.")


@router.post("/ask")
async def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    """RAG-based question answering with citations and conversation memory."""
    try:
        result = rag_qa.answer_question(
            query=request.query,
            session_id=request.session_id,
            doc_ids=request.doc_ids,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer. Check GROQ_API_KEY is set.")

    # Log the query for analytics (4.10 — most-queried docs, total questions answered)
    doc_ids_referenced = ",".join(sorted(set(c["document"] for c in result["citations"])))
    log_entry = QueryLog(
        session_id=request.session_id,
        question=request.query,
        doc_ids_referenced=doc_ids_referenced,
        timestamp=datetime.datetime.utcnow(),
    )
    db.add(log_entry)
    db.commit()

    return result