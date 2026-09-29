

import hashlib
import json
import os
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


# ---------- Retrieval ----------

def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the top matching chunks with their citation metadata."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty question.")

    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer.")

    chunks, index = load_data()
    k = min(k, len(chunks))

    question_vector = embed_question(question)
    scores, indices = index.search(question_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score),
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