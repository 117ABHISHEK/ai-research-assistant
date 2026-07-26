import os
os.environ["USE_TF"] = "0"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import logging
from fastapi import FastAPI

from src.database.base import init_db
from routes import document_routes, search_routes, analysis_routes, analytics_routes

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Research & Knowledge Assistant")

init_db()

app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)


@app.get("/")
async def root():
    return {"message": "AI Research & Knowledge Assistant API is running."}
