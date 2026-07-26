from groq import Groq
from typing import Dict, Any
import logging

from config.settings import settings
from src.vector_store.manager import vector_store_manager

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_TEMPLATE = """You are an AI Research Assistant. Summarize the following document content.

Document Content:
{context}

Produce your response in exactly this structure:

EXECUTIVE SUMMARY:
(2-3 sentence high-level overview for a non-technical reader)

TECHNICAL SUMMARY:
(A more detailed summary covering methods, approach, or technical specifics)

BULLET POINT SUMMARY:
- (key point 1)
- (key point 2)
- (key point 3)
(as many as relevant)

KEY TAKEAWAYS:
- (most important takeaway 1)
- (most important takeaway 2)

Only use information present in the document content above. Do not invent details."""


class DocumentSummarizer:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def summarize_document(self, doc_id: str, file_name: str = None, max_chunks: int = 20) -> Dict[str, Any]:
        """Fetches all chunks for a doc_id and generates a structured multi-section summary."""

        # Retrieve chunks belonging to this doc via a broad query, filtered by doc_id
        all_chunks = vector_store_manager.collection.get(where={"doc_id": doc_id})

        if not all_chunks["documents"]:
            return {
                "doc_id": doc_id,
                "summary": "No content found for this document. Ensure it has been processed successfully.",
            }

        texts = all_chunks["documents"][:max_chunks]  # cap to avoid overly long prompts
        combined_context = "\n\n".join(texts)

        prompt = SUMMARY_PROMPT_TEMPLATE.format(context=combined_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            summary_text = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Summarization failed for {doc_id}: {e}")
            raise

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "summary": summary_text,
            "chunks_used": len(texts),
        }


# Singleton
summarizer = DocumentSummarizer()