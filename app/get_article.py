
import json
import re
import sys
from functools import lru_cache

from quote_exact import load_pages


# ---------- Configuration ----------

SOURCE = "eu_ai_act_2024.pdf"
FIRST_ARTICLE = 1
LAST_ARTICLE = 113
PART_CHARS = 6000          # long Articles are returned in parts

HEADING = re.compile(r"Article\s+(\d+)")
# Structural headings where an Article's text ends.
STOP = re.compile(r"(CHAPTER\s+[IVXLC]+|SECTION\s+\d+|ANNEX\s+[IVXLC]+|Done at\b)")


# ---------- Build the Article index once ----------

@lru_cache(maxsize=1)
def load_articles() -> dict:
    """Map each Article number to its title, pages and text blocks."""
    pages = load_pages()
    blocks = []  # (page, block text) in reading order

    for (source, page), text in sorted(pages.items()):
        if source != SOURCE:
            continue
        for block in text.split("\n\n"):
            block = block.strip()
            if block:
                blocks.append((page, block))

    # Headings are blocks that are exactly "Article N", in increasing order.
    # (This skips mentions like "Article 49" inside the Annexes.)
    starts = []
    expected = FIRST_ARTICLE
    for index, (_, block) in enumerate(blocks):
        match = HEADING.fullmatch(block)
        if match and int(match.group(1)) == expected:
            starts.append((expected, index))
            expected += 1
            if expected > LAST_ARTICLE:
                break

    articles = {}
    for position, (number, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(blocks)

        body = []
        for page, block in blocks[start + 2:end]:   # skip heading + title
            if STOP.match(block):
                break
            body.append((page, block))

        title = blocks[start + 1][1].strip("`' ") if start + 1 < len(blocks) else ""
        pages_used = sorted({blocks[start][0]} | {page for page, _ in body})

        articles[number] = {"title": title, "pages": pages_used, "body": body}

    return articles


# ---------- Paragraph / point selection ----------

def select_paragraph(body: list, number: int) -> list | None:
    """Return the blocks of paragraph 'N.' or, if none, point '(N)'."""
    for style in (r"{n}\.\s", r"\({n}\)\s"):
        start_pattern = re.compile(style.format(n=number))
        next_pattern = re.compile(style.format(n=number + 1))

        start = next((i for i, (_, b) in enumerate(body)
                      if start_pattern.match(b)), None)
        if start is None:
            continue

        end = next((i for i in range(start + 1, len(body))
                    if next_pattern.match(body[i][1])), len(body))
        return body[start:end]

    return None


# ---------- get_article tool ----------

def get_article(article: int, paragraph: int | None = None,
                part: int = 1) -> dict:
    """Return an EU AI Act Article (or one paragraph/point of it)."""
    if type(article) is not int or not FIRST_ARTICLE <= article <= LAST_ARTICLE:
        return {"ok": False,
                "error": f"Article must be an integer from {FIRST_ARTICLE} to {LAST_ARTICLE}."}

    if paragraph is not None and (type(paragraph) is not int or paragraph < 1):
        return {"ok": False, "error": "Paragraph must be a positive integer."}

    if type(part) is not int or part < 1:
        return {"ok": False, "error": "Part must be a positive integer."}

    try:
        articles = load_articles()
    except (OSError, ValueError) as error:
        return {"ok": False, "error": f"Cannot load corpus: {error}"}

    if article not in articles:
        return {"ok": False, "error": f"Article {article} was not found in the corpus."}

    entry = articles[article]
    body = entry["body"]

    if paragraph is not None:
        body = select_paragraph(body, paragraph)
        if body is None:
            return {
                "ok": False,
                "error": f"Article {article} has no paragraph {paragraph}. "
                         "Call again without a paragraph to read it all.",
            }

    # Split long Articles into parts between paragraphs, never mid-sentence.
    groups = [[]]
    size = 0
    for page, block in body:
        if groups[-1] and size + len(block) > PART_CHARS:
            groups.append([])
            size = 0
        groups[-1].append((page, block))
        size += len(block) + 2

    parts = len(groups)
    if part > parts:
        return {"ok": False, "error": f"Only {parts} part(s) available."}

    selected = groups[part - 1]
    chunk = "\n\n".join(block for _, block in selected)
    label = f"Article {article}" + (f"({paragraph})" if paragraph else "")

    return {
        "ok": True,
        "source": SOURCE,
        "article": article,
        "paragraph": paragraph,
        "reference": label,
        "title": entry["title"],
        "pages": sorted({page for page, _ in selected}) or entry["pages"],
        "part": part,
        "parts": parts,
        "text": chunk,
        "note": (f"Part {part} of {parts}; call again with part={part + 1} "
                 "for the rest." if part < parts else "Complete text."),
    }


# ---------- Tool schema for the agents ----------

GET_ARTICLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_article",
        "description": (
            "Return the exact, complete text of an EU AI Act Article by "
            "number (1 to 113), with its title and pages. Optionally one "
            "numbered paragraph, e.g. article=99, paragraph=3 for Article "
            "99(3), or one definition point, e.g. article=3, paragraph=4. "
            "Use it whenever the question names an Article, instead of "
            "relying on search excerpts. EU AI Act only: never use it for "
            "NIST AI RMF questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "article": {"type": "integer",
                            "description": "Article number, 1 to 113."},
                "paragraph": {"type": "integer",
                              "description": "Optional paragraph or point number."},
                "part": {"type": "integer",
                         "description": "For long Articles: which part (default 1)."},
            },
            "required": ["article"],
            "additionalProperties": False,
        },
    },
}


# ---------- Run from the terminal ----------

def main() -> None:
    """Usage: python app/get_article.py ARTICLE [PARAGRAPH]"""
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print("Usage: python app/get_article.py ARTICLE [PARAGRAPH]   e.g. 99 3")
        return

    article = int(sys.argv[1])
    paragraph = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    print(json.dumps(get_article(article, paragraph), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
