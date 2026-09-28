import json
from pathlib import Path
from typing import TypedDict


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------- Data structure ----------

class Chunk(TypedDict):
    """A text passage and its original source location."""

    text: str
    source: str
    page: int


# ---------- Chunking ----------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop once the final characters have been included.
        if end == len(text):
            break

        start += chunk_size - overlap

    return chunks


def chunk_corpus(corpus: list[dict]) -> list[Chunk]:
    """Chunk each page separately, preserving filename and page number."""
    chunks: list[Chunk] = []

    for entry in corpus:
        for text in chunk_text(entry["text"]):
            chunks.append(
                {
                    "text": text,
                    "source": entry["source"],
                    "page": entry["page"],
                }
            )

    return chunks


# ---------- Main workflow ----------

def main() -> None:
    """Read the corpus, create chunks, and save them as UTF-8 JSON."""
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks = chunk_corpus(corpus)

    if not chunks:
        raise ValueError("The corpus contains no text to chunk.")

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Corpus entries: {len(corpus)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Overlap: {CHUNK_OVERLAP} characters")
    print(f"Total chunks: {len(chunks)}")
    print("Saved: data/chunks.json")


if __name__ == "__main__":
    main()