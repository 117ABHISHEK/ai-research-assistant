import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pytest
from src.ml.predictor import document_classifier


def test_classifier_model_loads():
    assert document_classifier.model is not None
    assert document_classifier.label_to_id is not None
    assert len(document_classifier.label_to_id) == 7


def test_classifier_returns_valid_category():
    result = document_classifier.predict(
        "This paper presents a convolutional neural network for image segmentation."
    )
    assert result["category"] in document_classifier.label_to_id
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["all_scores"]) == 7
