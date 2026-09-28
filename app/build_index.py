import hashlib
import json
import os
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
CACHE_FILE = DATA_DIR / "embeddings.npy"
METADATA_FILE = DATA_DIR / "index_metadata.json"
 
MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 50
 
 
# ---------- Load input ----------
 
def load_chunks() -> tuple[list[dict], str]:
    """Read chunks and calculate a fingerprint to detect changes."""
    raw_data = CHUNKS_FILE.read_bytes()
    chunks = json.loads(raw_data.decode("utf-8"))
 
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list.")
 
    for chunk in chunks:
        if not isinstance(chunk.get("text"), str):
            raise ValueError("Every chunk must contain text.")
        if not chunk["text"].strip():
            raise ValueError("A chunk contains empty text.")
 
    fingerprint = hashlib.sha256(raw_data).hexdigest()
    return chunks, fingerprint
 
 
def create_client() -> OpenAI:
    """Load the API key from the project's .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
 
    if not key or key.startswith(("your_", "replace_")):
        raise ValueError("Add your actual API key to .env.")
 
    return OpenAI(api_key=key)
 
 
# ---------- Protect saved work ----------
 
def check_metadata(fingerprint: str) -> None:
    """Check that saved embeddings belong to these exact chunks."""
    expected = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "chunks_sha256": fingerprint,
    }
 
    if METADATA_FILE.exists():
        saved = json.loads(
            METADATA_FILE.read_text(encoding="utf-8")
        )
 
        if saved != expected:
            raise ValueError(
                "Chunks or model settings changed. Move the old "
                "index_metadata.json, embeddings.npy and my_index.faiss "
                "out of data/ before rebuilding."
            )
    else:
        if CACHE_FILE.exists() or INDEX_FILE.exists():
            raise ValueError(
                "An index or embedding cache exists without metadata. "
                "Move those old files out of data/ before rebuilding."
            )
 
        METADATA_FILE.write_text(
            json.dumps(expected, indent=2) + "\n",
            encoding="utf-8",
        )
 
 
def save_progress(vectors: np.ndarray) -> None:
    """Save completed embeddings so interrupted work can resume."""
    temporary_file = CACHE_FILE.with_suffix(".tmp")
 
    with temporary_file.open("wb") as file:
        np.save(file, vectors, allow_pickle=False)
 
    temporary_file.replace(CACHE_FILE)
 
 
# ---------- Generate embeddings ----------
 
def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Create embeddings in batches, reusing saved progress."""
    if CACHE_FILE.exists():
        vectors = np.load(CACHE_FILE, allow_pickle=False)
    else:
        vectors = np.empty((0, DIMENSIONS), dtype=np.float32)
 
    if (
        vectors.ndim != 2
        or vectors.shape[1] != DIMENSIONS
        or len(vectors) > len(chunks)
    ):
        raise ValueError("The saved embedding cache has an invalid shape.")
 
    if len(vectors) == len(chunks):
        print("All embeddings already saved.")
        return vectors
 
    client = create_client()
    print(f"Previously embedded: {len(vectors)}/{len(chunks)}")
 
    for start in range(len(vectors), len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
 
        response = client.embeddings.create(
            model=MODEL,
            input=[chunk["text"] for chunk in batch],
            encoding_format="float",
        )
 
        # Preserve the same order as the chunks.
        items = sorted(response.data, key=lambda item: item.index)
        batch_vectors = np.asarray(
            [item.embedding for item in items],
            dtype=np.float32,
        )
 
        if batch_vectors.shape != (len(batch), DIMENSIONS):
            raise ValueError("Unexpected embedding dimensions.")
 
        vectors = np.concatenate((vectors, batch_vectors), axis=0)
 
        save_progress(vectors)
        track_cost(response, is_embedding=True)
 
        print(f"Embedded: {len(vectors)}/{len(chunks)}")
 
    return vectors
 
 
# ---------- Build index ----------
 
def main() -> None:
    """Load chunks, create embeddings, and save the search index."""
    chunks, fingerprint = load_chunks()
    check_metadata(fingerprint)
 
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
 
        if index.ntotal != len(chunks) or index.d != DIMENSIONS:
            raise ValueError("The existing index has an unexpected size.")
 
        print(f"Index already exists: {index.ntotal} vectors.")
        print("No API calls made.")
        return
 
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {MODEL}")
 
    vectors = embed_chunks(chunks)
 
    # Normalize vectors so inner-product search uses cosine similarity.
    faiss.normalize_L2(vectors)
 
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(vectors)
 
    temporary_file = INDEX_FILE.with_suffix(".tmp")
    faiss.write_index(index, str(temporary_file))
    temporary_file.replace(INDEX_FILE)
 
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")
    print("Saved: data/my_index.faiss")
 
 
if __name__ == "__main__":
    main()