from groq import Groq
from typing import List, Dict, Any
import logging

from config.settings import settings
from src.vector_store.manager import vector_store_manager

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are an AI Research Assistant. Answer the user's question using ONLY the provided document context below.
If the context does not contain sufficient information to answer, respond exactly with: "I cannot determine the answer from the provided documents."

Conversation History:
{history}

Context:
{context}

Question: {question}

Provide a clear, direct answer. Do not invent information that is not present in the context above."""


class RAGQuestionAnswering:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        # In-memory conversation store: {session_id: [{"role": ..., "content": ...}, ...]}
        self._memory: Dict[str, List[Dict[str, str]]] = {}

    def _get_history_text(self, session_id: str, max_turns: int = 5) -> str:
        history = self._memory.get(session_id, [])
        recent = history[-max_turns:]
        if not recent:
            return "(no prior conversation)"
        return "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent)

    def _update_memory(self, session_id: str, question: str, answer: str):
        self._memory.setdefault(session_id, [])
        self._memory[session_id].append({"role": "user", "content": question})
        self._memory[session_id].append({"role": "assistant", "content": answer})

    def answer_question(
        self,
        query: str,
        session_id: str = "default",
        doc_ids: List[str] = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """Retrieves relevant context and generates a citation-grounded response."""

        retrieved = vector_store_manager.semantic_search(query, top_k=top_k, doc_ids=doc_ids)

        if not retrieved:
            answer = "I cannot determine the answer from the provided documents."
            self._update_memory(session_id, query, answer)
            return {
                "answer": answer,
                "citations": [],
                "retrieved_context": [],
                "confidence": 0.0,
            }

        context_str = ""
        citations = []
        for r in retrieved:
            doc_name = r["metadata"].get("file_name", "Unknown")
            page_no = r["metadata"].get("page_number", "N/A")
            context_str += f"\n--- Source: {doc_name} (Page {page_no}) ---\n{r['text']}\n"
            citations.append({"document": doc_name, "page": page_no, "chunk_id": r["chunk_id"]})

        history_text = self._get_history_text(session_id)

        prompt = PROMPT_TEMPLATE.format(
            history=history_text,
            context=context_str,
            question=query,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

        self._update_memory(session_id, query, answer)

        # Simple confidence proxy: inverse of average vector distance (lower distance = higher confidence)
        avg_distance = sum(r["distance"] for r in retrieved) / len(retrieved)
        confidence = max(0.0, min(1.0, 1 - (avg_distance / 2)))

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": [r["text"] for r in retrieved],
            "confidence": round(confidence, 3),
        }


# Lazy singleton — avoids initializing the Groq client at import time.
_rag_qa_instance = None

def get_rag_qa():
    global _rag_qa_instance
    if _rag_qa_instance is None:
        _rag_qa_instance = RAGQuestionAnswering()
    return _rag_qa_instance


class _LazyRagQA:
    def __getattr__(self, name):
        return getattr(get_rag_qa(), name)


rag_qa = _LazyRagQA()
