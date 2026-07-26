from groq import Groq
from typing import Dict, Any, List
import logging

from config.settings import settings
from src.vector_store.manager import vector_store_manager

logger = logging.getLogger(__name__)

COMPARISON_PROMPT_TEMPLATE = """You are an AI Research Assistant. Compare the following documents using ONLY the content provided.

{documents_block}

Produce your response in exactly this structure:

METHODOLOGIES:
(Compare the approaches/methods used in each document)

ADVANTAGES & DISADVANTAGES:
(Compare strengths and weaknesses across documents)

SIMILARITIES:
(What do the documents have in common)

DIFFERENCES:
(How do the documents differ)

IMPLEMENTATION APPROACHES:
(Compare any technical/implementation details mentioned)

Only use information present in the documents above. If a document lacks information for a section, state that clearly rather than inventing details."""


class DocumentComparator:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def compare_documents(self, doc_ids: List[str], max_chunks_per_doc: int = 15) -> Dict[str, Any]:
        """Retrieves content for multiple doc_ids and generates a structured comparison."""

        if len(doc_ids) < 2:
            raise ValueError("At least 2 doc_ids are required for comparison.")

        documents_block = ""
        for doc_id in doc_ids:
            chunks = vector_store_manager.collection.get(where={"doc_id": doc_id})
            if not chunks["documents"]:
                documents_block += f"\n--- Document: {doc_id} (NO CONTENT FOUND) ---\n"
                continue

            texts = chunks["documents"][:max_chunks_per_doc]
            file_name = chunks["metadatas"][0].get("file_name", doc_id) if chunks["metadatas"] else doc_id
            combined = "\n".join(texts)
            documents_block += f"\n--- Document: {file_name} (doc_id={doc_id}) ---\n{combined}\n"

        prompt = COMPARISON_PROMPT_TEMPLATE.format(documents_block=documents_block)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            comparison_text = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Comparison failed for {doc_ids}: {e}")
            raise

        return {
            "doc_ids": doc_ids,
            "comparison": comparison_text,
        }


# Singleton
comparator = DocumentComparator()