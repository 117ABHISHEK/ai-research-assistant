import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import tensorflow as tf
import pickle
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Loads the trained TensorFlow model and label mapping for inference on new documents."""

    def __init__(self):
        self.model = None
        self.label_to_id = None
        self.id_to_label = None
        self._load()

    def _load(self):
        try:
            self.model = tf.keras.models.load_model(settings.TF_MODEL_PATH)
            with open(settings.TOKENIZER_PATH, "rb") as f:
                mapping = pickle.load(f)
                self.label_to_id = mapping["label_to_id"]
                self.id_to_label = mapping["id_to_label"]
            logger.info("Document classifier model and label mapping loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            self.model = None

    def predict(self, text: str) -> dict:
        """Classifies a document's text into one of the trained categories."""
        if self.model is None:
            return {"category": "Unknown", "confidence": 0.0, "error": "Model not loaded"}

        # Truncate very long text — model was trained on abstract-length text (~200 tokens)
        truncated = text[:3000]

        input_tensor = tf.constant([truncated], dtype=tf.string)
        predictions = self.model.predict(input_tensor, verbose=0)[0]

        predicted_id = int(predictions.argmax())
        confidence = float(predictions[predicted_id])
        category = self.id_to_label.get(predicted_id, "Unknown")

        return {
            "category": category,
            "confidence": round(confidence, 3),
            "all_scores": {
                self.id_to_label[i]: round(float(score), 3)
                for i, score in enumerate(predictions)
            },
        }


# Singleton instance — loaded once at app startup, reused across requests
document_classifier = DocumentClassifier()