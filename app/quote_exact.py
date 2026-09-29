"""Return the exact stored text of a document page from corpus.json."""

import argparse
import json
from functools import lru_cache
from pathlib import Path


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = PROJECT_ROOT / "data" / "corpus.json"


# ---------- Load corpus ----------

@lru_cache(maxsize=1)
def load_pages() -> dict:
    """Load the corpus once per process and index it by source and page."""
    entries = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))

    if not isinstance(entries, list) or not entries:
        raise ValueError("The corpus must contain a non-empty list.")

    pages = {}

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each corpus entry must be an object.")

        source = entry.get("source")
        page = entry.get("page")
        text = entry.get("text")

        if not isinstance(source, str) or not source:
            raise ValueError("A corpus entry has an invalid source.")

        if type(page) is not int or page < 1:
            raise ValueError("A corpus entry has an invalid page number.")

        if not isinstance(text, str):
            raise ValueError("A corpus entry has invalid text.")

        key = (source, page)

        if key in pages:
            raise ValueError(f"Duplicate corpus page: {source}, page {page}.")

        pages[key] = text

    return pages


# ---------- Quote tool ----------

def quote_exact(source: str, page: int) -> dict:
    """Return a complete stored page without rewriting its text."""
    if not isinstance(source, str) or not source.strip():
        return {"ok": False, "error": "Provide a document filename."}

    if type(page) is not int or page < 1:
        return {"ok": False, "error": "Page must be a positive integer."}

    source = source.strip()

    try:
        pages = load_pages()
    except FileNotFoundError:
        return {"ok": False, "error": "data/corpus.json was not found."}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": f"Cannot load corpus: {error}"}

    available_sources = sorted({name for name, _ in pages})

    if source not in available_sources:
        return {
            "ok": False,
            "error": "Document not found. Use an exact filename.",
            "available_sources": available_sources,
        }

    key = (source, page)

    if key not in pages:
        available_pages = sorted(
            number for name, number in pages if name == source
        )
        return {
            "ok": False,
            "error": "Page not found in this document.",
            "source": source,
            "available_pages": available_pages,
        }

    text = pages[key]

    return {
        "ok": True,
        "source": source,
        "page": page,
        "text": text,
        "char_count": len(text),
        "empty": not text.strip(),
    }


# ---------- Run from terminal ----------

def main() -> None:
    """Read the document filename and page number from the terminal."""
    parser = argparse.ArgumentParser(
        description="Return a full page from the extracted corpus."
    )
    parser.add_argument("source", help="Exact document filename")
    parser.add_argument("page", type=int, help="PDF page number, starting at 1")
    args = parser.parse_args()

    result = quote_exact(args.source, args.page)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()