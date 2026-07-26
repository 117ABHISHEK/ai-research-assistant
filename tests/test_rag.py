import pytest
from src.vector_store.manager import vector_store_manager


def test_vector_store_indexes_and_retrieves():
    test_chunks = [
        {
            "chunk_id": "pytest-test-doc_c0",
            "doc_id": "pytest-test-doc",
            "page_number": 1,
            "text": "The quick brown fox jumps over the lazy dog.",
        }
    ]
    vector_store_manager.embed_and_index_chunks(test_chunks, file_name="pytest_test.pdf")

    results = vector_store_manager.semantic_search("a fox jumping", top_k=1, doc_ids=["pytest-test-doc"])

    assert len(results) == 1
    assert results[0]["metadata"]["doc_id"] == "pytest-test-doc"

    vector_store_manager.delete_document_chunks("pytest-test-doc")


def test_keyword_search_finds_exact_match():
    test_chunks = [
        {
            "chunk_id": "pytest-kw-doc_c0",
            "doc_id": "pytest-kw-doc",
            "page_number": 1,
            "text": "UNIQUE_TEST_TOKEN_12345 appears in this text.",
        }
    ]
    vector_store_manager.embed_and_index_chunks(test_chunks, file_name="pytest_kw.pdf")

    results = vector_store_manager.keyword_search("UNIQUE_TEST_TOKEN_12345", doc_ids=["pytest-kw-doc"])

    assert len(results) == 1
    assert "UNIQUE_TEST_TOKEN_12345" in results[0]["text"]

    vector_store_manager.delete_document_chunks("pytest-kw-doc")
