from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- App ---
    APP_NAME: str = "AI Research & Knowledge Assistant"
    DEBUG: bool = True

    # --- LLM (Groq) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths ---
    RAW_DOCUMENTS_DIR: str = "./data/raw_documents"
    VECTOR_DB_DIR: str = "./data/vector_db"
    DATASET_DIR: str = "./data/dataset"

    # --- ML model paths ---
    TF_MODEL_PATH: str = "./models/tf_classifier.keras"
    TOKENIZER_PATH: str = "./models/tokenizer.pickle"

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # --- Chunking ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # --- Retrieval ---
    TOP_K: int = 4


settings = Settings()