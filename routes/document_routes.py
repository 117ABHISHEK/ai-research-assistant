from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
import os
import uuid
import datetime
import logging

from src.database.base import get_db
from src.database.models import Document
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.vector_store.manager import vector_store_manager
from src.ml.predictor import document_classifier
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Management"])

os.makedirs(settings.RAW_DOCUMENTS_DIR, exist_ok=True)

parser = PDFParser()
chunker = DocumentChunker()


def process_pdf_pipeline(doc_id: str, file_path: str, file_name: str, db: Session):
    """Background pipeline: extract -> classify -> chunk -> embed -> index -> update status."""
    doc_record = db.query(Document).filter(Document.doc_id == doc_id).first()
    try:
        # 1. Extract text with page metadata
        pages = parser.extract_text_with_metadata(file_path, doc_id)
        total_pages = parser.get_page_count(file_path)

        if not pages:
            raise ValueError("No extractable text found in PDF (possibly scanned/image-only).")

        # 2. Classify document using full extracted text (first ~3000 chars, handled inside predictor)
        full_text = " ".join(p["text"] for p in pages)
        classification = document_classifier.predict(full_text)

        # 3. Chunk text
        chunks = chunker.create_chunks(pages)

        # 4. Embed and index into vector store
        vector_store_manager.embed_and_index_chunks(chunks, file_name=file_name)

        # 5. Update metadata record
        doc_record.total_pages = total_pages
        doc_record.total_chunks = len(chunks)
        doc_record.category = classification["category"]
        doc_record.processing_status = "PROCESSED"
        db.commit()

        logger.info(f"Successfully processed document {doc_id}: {len(chunks)} chunks, category={classification['category']}")

    except Exception as e:
        logger.error(f"Processing failed for document {doc_id}: {e}")
        doc_record.processing_status = "FAILED"
        db.commit()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads a PDF document and triggers background processing pipeline."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(settings.RAW_DOCUMENTS_DIR, f"{doc_id}_{file.filename}")

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    doc_record = Document(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=file_path,
        upload_timestamp=datetime.datetime.utcnow(),
        processing_status="PROCESSING",
    )
    db.add(doc_record)
    db.commit()

    background_tasks.add_task(process_pdf_pipeline, doc_id, file_path, file.filename, db)

    return {
        "message": "Document uploaded successfully. Processing started.",
        "doc_id": doc_id,
        "file_name": file.filename,
        "status": "PROCESSING",
    }


@router.get("/")
async def list_documents(db: Session = Depends(get_db)):
    """Lists all uploaded documents with metadata."""
    docs = db.query(Document).all()
    return [
        {
            "doc_id": d.doc_id,
            "file_name": d.file_name,
            "upload_timestamp": d.upload_timestamp,
            "total_pages": d.total_pages,
            "total_chunks": d.total_chunks,
            "processing_status": d.processing_status,
            "category": d.category,
        }
        for d in docs
    ]
@router.get("/{doc_id}")
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Retrieves metadata for a single document."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "doc_id": doc.doc_id,
        "file_name": doc.file_name,
        "upload_timestamp": doc.upload_timestamp,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "processing_status": doc.processing_status,
        "category": doc.category,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Deletes a document's file, vector chunks, and metadata record."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    vector_store_manager.delete_document_chunks(doc_id)

    db.delete(doc)
    db.commit()

    return {"message": f"Document {doc_id} deleted successfully."}


@router.post("/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Re-runs the processing pipeline on an already-uploaded document."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=400, detail="Original file no longer exists on disk.")

    vector_store_manager.delete_document_chunks(doc_id)
    doc.processing_status = "PROCESSING"
    db.commit()

    background_tasks.add_task(process_pdf_pipeline, doc_id, doc.file_path, doc.file_name, db)

    return {"message": f"Reprocessing started for document {doc_id}."}