import os
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import pickle
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def load_dataset(path: str = None) -> pd.DataFrame:
    path = path or f"{settings.DATASET_DIR}/training_data.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df


def build_and_train_classifier(
    train_texts,
    train_labels_encoded,
    num_classes: int,
    vocab_size: int = 10000,
    max_len: int = 200,
    epochs: int = 10,
):
    # Shuffle before splitting — dataset is grouped by category (80 rows per label in sequence),
    # so an unshuffled validation_split would validate on near-single-category data.
    import numpy as np
    indices = np.arange(len(train_texts))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_texts = tf.constant([train_texts[i] for i in indices], dtype=tf.string)
    train_labels_encoded = tf.constant([train_labels_encoded[i] for i in indices], dtype=tf.int32)

    # 1. Vectorization Layer
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorize_layer.adapt(train_texts)

    # 2. Neural Network Architecture
    model = models.Sequential([
        vectorize_layer,
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    # 3. Compilation
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Training
    history = model.fit(
        train_texts,
        train_labels_encoded,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # 5. Persistence
    model.save(settings.TF_MODEL_PATH)
    logger.info(f"Model saved to {settings.TF_MODEL_PATH}")

    return model, history


def run_training_pipeline():
    logging.basicConfig(level=logging.INFO)

    df = load_dataset()
    logger.info(f"Loaded {len(df)} rows for training")

    labels = sorted(df["label"].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    df["label_id"] = df["label"].map(label_to_id)

    model, history = build_and_train_classifier(
        train_texts=df["text"].tolist(),
        train_labels_encoded=df["label_id"].tolist(),
        num_classes=len(labels),
    )

    with open(settings.TOKENIZER_PATH, "wb") as f:
        pickle.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)
    logger.info(f"Label mapping saved to {settings.TOKENIZER_PATH}")

    final_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    logger.info(f"Final train accuracy: {final_acc:.3f} | val accuracy: {final_val_acc:.3f}")

    return model, label_to_id


if __name__ == "__main__":
    run_training_pipeline()