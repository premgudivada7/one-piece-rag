

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from cost_tracker import track_cost


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE = DATA_DIR / "my_index.faiss"
METADATA_FILE = DATA_DIR / "index_metadata.json"

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
TOP_K = 5

# "hybrid" = vector search + BM25 keyword search, merged with Reciprocal
# Rank Fusion (RRF). "vector" = embeddings only (the original behaviour).
# Hybrid helps with exact terms (Article numbers, GOVERN/MAP/...) and stops
# short title chunks ("NIST AI RMF 1.0") from crowding out the real answer.
SEARCH_MODE = "hybrid"
RRF_K = 60            # standard RRF constant: 1 / (60 + rank)

# Weight of each ranking in the fusion (vector ranking weight = 1).
# Tuned offline with the real question vectors (scripts/save_question_vectors.py):
# keyword 0.5-0.8 and Article 0.5-1.5 all reach 33/33 hits@5 on questions.json;
# 0.7 / 1.0 sit in the middle of that range. Keyword weight 1.0 missed #10,
# vector only missed 7 questions.
KEYWORD_WEIGHT = 0.7
ARTICLE_WEIGHT = 1.0

EU_SOURCE = "eu_ai_act_2024.pdf"


# ---------- Load data and client ----------

@lru_cache(maxsize=1)
def load_data():
    """Load and validate the matching chunks and index once."""
    raw_data = CHUNKS_FILE.read_bytes()
    chunks = json.loads(raw_data.decode("utf-8"))
    index = faiss.read_index(str(INDEX_FILE))

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list.")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be an object.")

        if not isinstance(chunk.get("text"), str):
            raise ValueError("A chunk has missing or invalid text.")

        if not isinstance(chunk.get("source"), str):
            raise ValueError("A chunk has missing or invalid source metadata.")

        if type(chunk.get("page")) is not int or chunk["page"] < 1:
            raise ValueError("A chunk has an invalid page number.")

    if index.ntotal != len(chunks):
        raise ValueError("The index and chunks have different sizes.")

    if index.d != DIMENSIONS:
        raise ValueError("The index has unexpected embedding dimensions.")

    # Verify the fingerprint saved by our index builder, when available.
    if METADATA_FILE.exists():
        metadata = json.loads(
            METADATA_FILE.read_text(encoding="utf-8")
        )

        expected = {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "chunks_sha256": hashlib.sha256(raw_data).hexdigest(),
        }

        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "The index metadata does not match these chunks or model. "
                "Download the matching files shared by your team."
            )

    return chunks, index


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create one reusable API client."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()

    if not key or key.startswith(("your_", "replace_")):
        raise ValueError("Add your actual OPENAI_API_KEY to .env.")

    return OpenAI(api_key=key)


# ---------- Question embedding ----------

def embed_question(question: str) -> np.ndarray:
    """Embed a question, track its cost, and normalize its vector."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty question.")

    response = get_client().embeddings.create(
        model=MODEL,
        input=question.strip(),
        dimensions=DIMENSIONS,
        encoding_format="float",
    )

    track_cost(response, is_embedding=True)

    vector = np.asarray(
        [response.data[0].embedding],
        dtype=np.float32,
    )

    if vector.shape != (1, DIMENSIONS) or not np.isfinite(vector).all():
        raise ValueError("The API returned an invalid question vector.")

    faiss.normalize_L2(vector)
    return vector


# ---------- Keyword search (BM25) ----------

def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, e.g. 'Article 99(3)' -> article, 99, 3."""
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def load_bm25():
    """Build the BM25 keyword index once, or None if rank_bm25 is missing."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 is not installed: using vector search only. "
              "Install it with: uv pip install rank-bm25")
        return None

    chunks, _ = load_data()
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def article_pages(question: str) -> set[tuple[str, int]]:
    """Pages of every EU AI Act Article named in the question.

    "How does Article 3 define risk?" -> the pages of Article 3. Chunks
    rarely contain their Article heading, so this helps retrieval find
    the right Article. Uses the heading index built by get_article.py.
    """
    numbers = [int(n) for n in re.findall(r"\bArticle\s+(\d+)", question,
                                          flags=re.IGNORECASE)]
    if not numbers:
        return set()

    from get_article import load_articles  # local import: only when needed

    articles = load_articles()
    return {
        (EU_SOURCE, page)
        for number in numbers if number in articles
        for page in articles[number]["pages"]
    }


# ---------- Retrieval ----------

def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the top matching chunks with their citation metadata.

    Every chunk keeps its cosine similarity in "score" (used for the
    out-of-scope threshold and shown in the UI). In hybrid mode the order
    comes from RRF over the vector ranking and the BM25 ranking.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty question.")

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    chunks, index = load_data()
    k = min(k, len(chunks))

    question_vector = embed_question(question)

    # Rank ALL chunks by cosine similarity (1,776 vectors: instant).
    scores, indices = index.search(question_vector, len(chunks))
    vector_order = [int(i) for i in indices[0] if i >= 0]
    cosine = {int(i): float(s) for s, i in zip(scores[0], indices[0]) if i >= 0}

    bm25 = load_bm25() if SEARCH_MODE == "hybrid" else None

    if bm25 is None:
        order = vector_order[:k]
        keyword_rank = {}
        article_rank = {}
    else:
        keyword_scores = bm25.get_scores(tokenize(question))
        keyword_order = [
            i for i in np.argsort(-keyword_scores) if keyword_scores[i] > 0
        ]
        keyword_rank = {int(i): r for r, i in enumerate(keyword_order, 1)}

        # Third ranking: chunks on the pages of an Article the question
        # names, in vector order (empty when no Article is named).
        wanted = article_pages(question)
        article_order = [i for i in vector_order
                         if (chunks[i]["source"], chunks[i]["page"]) in wanted]
        article_rank = {i: r for r, i in enumerate(article_order, start=1)}

        fused = {}
        for rank, i in enumerate(vector_order, start=1):
            fused[i] = 1 / (RRF_K + rank)
        for i, rank in keyword_rank.items():
            fused[i] = fused.get(i, 0) + KEYWORD_WEIGHT / (RRF_K + rank)
        for i, rank in article_rank.items():
            fused[i] = fused.get(i, 0) + ARTICLE_WEIGHT / (RRF_K + rank)

        order = sorted(fused, key=fused.get, reverse=True)[:k]

    vector_rank = {i: r for r, i in enumerate(vector_order, start=1)}

    results = []
    for idx in order:
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": cosine[idx],
            "vector_rank": vector_rank.get(idx),
            "keyword_rank": keyword_rank.get(idx),
            "article_rank": article_rank.get(idx),
        })

    return results


# ---------- Terminal check ----------

if __name__ == "__main__":
    question = "What is a high-risk AI system?"
    results = retrieve(question)

    print(f"\nQuestion: {question}\n")

    for number, result in enumerate(results, start=1):
        print(f"--- Result {number} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}")
        print()