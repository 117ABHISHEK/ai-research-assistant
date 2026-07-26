import pytest
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker


def test_chunker_produces_overlapping_chunks():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    pages_data = [
        {"doc_id": "test-doc", "page_number": 1, "text": "A" * 250}
    ]
    chunks = chunker.create_chunks(pages_data)

    assert len(chunks) > 1
    assert all(c["doc_id"] == "test-doc" for c in chunks)
    assert all(c["page_number"] == 1 for c in chunks)
    assert chunks[0]["chunk_id"] == "test-doc_c0"


def test_chunker_handles_empty_pages():
    chunker = DocumentChunker()
    chunks = chunker.create_chunks([])
    assert chunks == []


def test_pdf_parser_cleans_text():
    parser = PDFParser()
    raw = "  Line one  \n\n\n   Line two   \n"
    cleaned = parser._clean_text(raw)
    assert "Line one" in cleaned
    assert "Line two" in cleaned
    assert "\n\n\n" not in cleaned
