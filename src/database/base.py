from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

from config.settings import settings

# Ensure the directory for the SQLite file exists
db_path = settings.DATABASE_URL.replace("sqlite:///", "")
os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once on app startup."""
    from src.database import models  # noqa: F401  (ensures models are registered)
    Base.metadata.create_all(bind=engine)