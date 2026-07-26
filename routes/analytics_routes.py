from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.analytics.metrics import get_analytics_summary

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary")
async def analytics_summary(db: Session = Depends(get_db)):
    """Returns system-wide analytics: document counts, chunk counts, category distribution, top-queried docs, total questions answered."""
    return get_analytics_summary(db)
