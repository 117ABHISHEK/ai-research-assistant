from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any

from src.database.models import Document, QueryLog


def get_analytics_summary(db: Session) -> Dict[str, Any]:
    total_documents = db.query(Document).count()
    total_chunks = db.query(func.sum(Document.total_chunks)).scalar() or 0
    total_questions_answered = db.query(QueryLog).count()

    category_counts = (
        db.query(Document.category, func.count(Document.doc_id))
        .filter(Document.category.isnot(None))
        .group_by(Document.category)
        .all()
    )
    category_distribution = {cat: count for cat, count in category_counts}

    status_counts = (
        db.query(Document.processing_status, func.count(Document.doc_id))
        .group_by(Document.processing_status)
        .all()
    )
    status_distribution = {status: count for status, count in status_counts}

    all_logs = db.query(QueryLog.doc_ids_referenced).filter(QueryLog.doc_ids_referenced != "").all()
    doc_query_counts: Dict[str, int] = {}
    for (refs,) in all_logs:
        if not refs:
            continue
        for name in refs.split(","):
            name = name.strip()
            if name:
                doc_query_counts[name] = doc_query_counts.get(name, 0) + 1

    top_queried = sorted(doc_query_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_documents": total_documents,
        "total_chunks": int(total_chunks),
        "total_embeddings_generated": int(total_chunks),
        "total_questions_answered": total_questions_answered,
        "category_distribution": category_distribution,
        "processing_status_distribution": status_distribution,
        "most_queried_documents": [{"document": doc, "query_count": count} for doc, count in top_queried],
    }
