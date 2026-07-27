import os
os.environ["USE_TF"] = "0"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        # Uses ONNX runtime instead of torch/transformers — same MiniLM model, much lighter import.
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
        )

    def embed_and_index_chunks(self, chunks: List[Dict[str, Any]], file_name: str) -> int:
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "file_name": file_name,
                "page_number": c["page_number"],
            }
            for c in chunks
        ]

        # Chroma computes embeddings internally via self.embedding_fn
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Indexed {len(chunks)} chunks for {file_name} into vector store.")
        return len(chunks)

    def semantic_search(self, query: str, top_k: int = None, doc_ids: List[str] = None) -> List[Dict[str, Any]]:
        top_k = top_k or settings.TOP_K

        where_filter = None
        if doc_ids:
            where_filter = {"doc_id": {"$in": doc_ids}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return matches

    def keyword_search(self, query: str, top_k: int = None, doc_ids: List[str] = None) -> List[Dict[str, Any]]:
        top_k = top_k or settings.TOP_K
        where_filter = None
        if doc_ids:
            where_filter = {"doc_id": {"$in": doc_ids}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}

        all_docs = self.collection.get(where=where_filter)
        query_lower = query.lower()

        matches = []
        for i, text in enumerate(all_docs["documents"]):
            if query_lower in text.lower():
                matches.append({
                    "chunk_id": all_docs["ids"][i],
                    "text": text,
                    "metadata": all_docs["metadatas"][i],
                })
        return matches[:top_k]

    def delete_document_chunks(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted all chunks for doc_id={doc_id}")


# Lazy singleton — avoids loading the embedding model at import time.
_vector_store_manager_instance = None

def get_vector_store_manager():
    global _vector_store_manager_instance
    if _vector_store_manager_instance is None:
        _vector_store_manager_instance = VectorStoreManager()
    return _vector_store_manager_instance


class _LazyVectorStoreManager:
    def __getattr__(self, name):
        return getattr(get_vector_store_manager(), name)


vector_store_manager = _LazyVectorStoreManager()
