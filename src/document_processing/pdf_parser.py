import fitz  # PyMuPDF
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Extracts text page-by-page from a PDF, preserving page number metadata."""

    def extract_text_with_metadata(self, pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise

        extracted_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text").strip()
            cleaned_text = self._clean_text(raw_text)
            if cleaned_text:
                extracted_pages.append({
                    "doc_id": doc_id,
                    "page_number": page_num + 1,
                    "text": cleaned_text,
                })

        total_pages = len(doc)
        doc.close()

        logger.info(f"Extracted {len(extracted_pages)} non-empty pages from {total_pages} total pages ({doc_id})")
        return extracted_pages

    def _clean_text(self, text: str) -> str:
        """Basic cleaning: collapse whitespace, strip stray control characters."""
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # drop empty lines
        return "\n".join(lines)

    def get_page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count