from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import logging

from src.database.base import get_db
from src.database.models import Document
from src.rag.summarizer import summarizer
from src.rag.comparator import comparator
from src.ml.predictor import document_classifier
from src.document_processing.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis: Summarize, Compare, Classify"])

parser = PDFParser()


class SummarizeRequest(BaseModel):
    doc_id: str


class CompareRequest(BaseModel):
    doc_ids: List[str]


class ClassifyRequest(BaseModel):
    doc_id: str


@router.post("/summarize")
async def summarize_document(request: SummarizeRequest, db: Session = Depends(get_db)):
    """Generates a multi-section summary (Executive, Technical, Bullet, Key Takeaways) for a document."""
    doc = db.query(Document).filter(Document.doc_id == request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.processing_status != "PROCESSED":
        raise HTTPException(status_code=400, detail=f"Document is not ready (status: {doc.processing_status}).")

    try:
        result = summarizer.summarize_document(doc_id=doc.doc_id, file_name=doc.file_name)
        return result
    except Exception as e:
        logger.error(f"Summarize endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Summarization failed.")


@router.post("/compare")
async def compare_documents(request: CompareRequest, db: Session = Depends(get_db)):
    """Compares 2+ documents across methodology, pros/cons, similarities, differences."""
    if len(request.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 doc_ids are required for comparison.")

    docs = db.query(Document).filter(Document.doc_id.in_(request.doc_ids)).all()
    if len(docs) != len(request.doc_ids):
        raise HTTPException(status_code=404, detail="One or more documents not found.")

    try:
        result = comparator.compare_documents(doc_ids=request.doc_ids)
        return result
    except Exception as e:
        logger.error(f"Compare endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Comparison failed.")


@router.post("/classify")
async def classify_document(request: ClassifyRequest, db: Session = Depends(get_db)):
    """Re-runs TensorFlow classification on a document's extracted text (on-demand classify endpoint)."""
    doc = db.query(Document).filter(Document.doc_id == request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        pages = parser.extract_text_with_metadata(doc.file_path, doc.doc_id)
        full_text = " ".join(p["text"] for p in pages)
        result = document_classifier.predict(full_text)

        doc.category = result["category"]
        db.commit()

        return result
    except Exception as e:
        logger.error(f"Classify endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed.")
