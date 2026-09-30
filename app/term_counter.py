

import json
import re
import sys

from quote_exact import load_pages


# ---------- Configuration ----------

MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 100
DEFAULT_MAX_PAGES = 10
SNIPPET_CHARS = 80


# ---------- Helpers ----------

def build_pattern(term: str) -> re.Pattern:
    """Case-insensitive match of the whole term, not part of a longer word.

    'risk' matches 'risk' and 'Risk' but not 'risks' or 'high-risk'.
    Spaces in the term match any whitespace (line breaks included).
    """
    words = [re.escape(word) for word in term.split()]
    body = r"\s+".join(words)
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.IGNORECASE)


def snippet(text: str, match: re.Match) -> str:
    """A short piece of text around the first match on a page."""
    start = max(0, match.start() - SNIPPET_CHARS)
    end = min(len(text), match.end() + SNIPPET_CHARS)
    piece = " ".join(text[start:end].split())
    return ("…" if start > 0 else "") + piece + ("…" if end < len(text) else "")


# ---------- count_term tool ----------

def count_term(term: str, source: str | None = None,
               max_pages: int = DEFAULT_MAX_PAGES) -> dict:
    """Count exact occurrences of a term per document and per page."""
    if not isinstance(term, str):
        return {"ok": False, "error": "Term must be a string."}

    term = " ".join(term.split())

    if len(term) < MIN_TERM_LENGTH:
        return {"ok": False, "error": "Term must have at least 2 characters."}

    if len(term) > MAX_TERM_LENGTH:
        return {"ok": False, "error": "Term is longer than 100 characters."}

    if type(max_pages) is not int or not 1 <= max_pages <= 50:
        return {"ok": False, "error": "max_pages must be an integer from 1 to 50."}

    try:
        pages = load_pages()
    except (OSError, ValueError) as error:
        return {"ok": False, "error": f"Cannot load corpus: {error}"}

    sources = sorted({name for name, _ in pages})

    if source is not None:
        source = source.strip()
        if source not in sources:
            return {
                "ok": False,
                "error": "Document not found. Use an exact filename.",
                "available_sources": sources,
            }

    pattern = build_pattern(term)
    by_document = {name: 0 for name in sources if source in (None, name)}
    hits = []

    for (name, page), text in sorted(pages.items()):
        if source is not None and name != source:
            continue

        matches = list(pattern.finditer(text))
        if not matches:
            continue

        by_document[name] += len(matches)
        hits.append({
            "source": name,
            "page": page,
            "count": len(matches),
            "snippet": snippet(text, matches[0]),
        })

    total = sum(by_document.values())

    # Pages with the most mentions first, then in document order.
    top_pages = sorted(hits, key=lambda h: (-h["count"], h["source"], h["page"]))

    return {
        "ok": True,
        "term": term,
        "total_count": total,
        "by_document": by_document,
        "pages_with_term": len(hits),
        "first_page": ({"source": hits[0]["source"], "page": hits[0]["page"]}
                       if hits else None),
        "top_pages": top_pages[:max_pages],
        "note": ("Exact, case-insensitive whole-term match."
                 if total else "The term does not appear in the corpus."),
    }


# ---------- Tool schema for the agents ----------

COUNT_TERM_TOOL = {
    "type": "function",
    "function": {
        "name": "count_term",
        "description": (
            "Count exact occurrences of a word or phrase in the documents "
            "(case-insensitive, whole term). Returns the total, the count per "
            "document, and the pages with the most mentions, each with a "
            "snippet. Use it for 'how many times / how many pages' questions "
            "and to locate exact terms such as 'Article 99' or 'GOVERN 1.2'. "
            "Call it ONCE without 'source' to get the count in each document "
            "AND the total (no addition needed). No meaning-based matching: "
            "use search_corpus for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The exact word or phrase, e.g. 'high-risk'.",
                },
                "source": {
                    "type": "string",
                    "enum": ["eu_ai_act_2024.pdf", "nist_ai_rmf_1_0.pdf"],
                    "description": "Optional: count in one document only.",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "How many top pages to list (1 to 50, default 10).",
                },
            },
            "required": ["term"],
            "additionalProperties": False,
        },
    },
}


# ---------- Run from the terminal ----------

def main() -> None:
    """Usage: python app/term_counter.py "term" [source]"""
    if len(sys.argv) < 2:
        print("Usage: python app/term_counter.py \"term\" [source.pdf]")
        return

    term = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(count_term(term, source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
