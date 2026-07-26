import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import time
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# arXiv category codes mapped to our 7 target labels
CATEGORY_MAP = {
    "Artificial Intelligence": "cs.AI",
    "Machine Learning": "cs.LG",
    "Computer Vision": "cs.CV",
    "Natural Language Processing": "cs.CL",
    "Robotics": "cs.RO",
    "Cyber Security": "cs.CR",
    "Cloud Computing": "cs.DC",  # distributed/cloud computing
}

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def fetch_abstracts_for_category(arxiv_code: str, max_results: int = 80) -> list:
    """Fetches paper abstracts for a given arXiv category code."""
    query = f"search_query=cat:{arxiv_code}&start=0&max_results={max_results}"
    url = f"{ARXIV_API_URL}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except Exception as e:
        logger.error(f"Failed to fetch from arXiv for {arxiv_code}: {e}")
        return []

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    abstracts = []
    for entry in root.findall("atom:entry", ns):
        summary = entry.find("atom:summary", ns)
        if summary is not None and summary.text:
            cleaned = " ".join(summary.text.strip().split())
            abstracts.append(cleaned)

    return abstracts


def build_dataset(max_per_category: int = 80) -> pd.DataFrame:
    """Builds a labelled DataFrame of (text, label) pairs across all categories."""
    rows = []

    for label, arxiv_code in CATEGORY_MAP.items():
        logger.info(f"Fetching abstracts for '{label}' ({arxiv_code})...")
        abstracts = fetch_abstracts_for_category(arxiv_code, max_results=max_per_category)
        for text in abstracts:
            rows.append({"text": text, "label": label})
        logger.info(f"  -> got {len(abstracts)} abstracts")
        time.sleep(3)  # be polite to arXiv's free API, avoid rate limiting

    df = pd.DataFrame(rows)
    logger.info(f"Total dataset size: {len(df)} rows across {df['label'].nunique()} categories")
    return df


def save_dataset(df: pd.DataFrame, path: str = None):
    path = path or f"{settings.DATASET_DIR}/training_data.csv"
    df.to_csv(path, index=False)
    logger.info(f"Dataset saved to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_dataset(max_per_category=80)
    save_dataset(df)
    print(df["label"].value_counts())