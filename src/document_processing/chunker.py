from typing import List, Dict, Any
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits page-level text into overlapping fixed-size chunks while
    preserving page number metadata for citation purposes.

    Strategy: fixed-size character chunking with overlap.
    Justification: simple, deterministic, and fast — overlap (150 chars)
    prevents context from being cut off mid-sentence at chunk boundaries,
    which matters for RAG retrieval accuracy. A more advanced approach
    (e.g. sentence-aware or semantic chunking) was considered but skipped
    given project time constraints; documented as a future improvement.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_id = 0

        for page in pages_data:
            text = page["text"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "chunk_id": f"{page['doc_id']}_c{chunk_id}",
                    "doc_id": page["doc_id"],
                    "page_number": page["page_number"],
                    "text": chunk_text,
                })

                chunk_id += 1
                start += (self.chunk_size - self.chunk_overlap)

        logger.info(f"Created {len(chunks)} chunks from {len(pages_data)} pages")
        return chunks