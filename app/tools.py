import difflib
import re
from datetime import date, datetime
from functools import lru_cache

from simpleeval import simple_eval

from app import config
from app.chunking import page_label
from app.index import get_index
from app.retrieval import best_score, is_confident, retrieve, score_name, threshold, unknown_words_note


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """A tool schema in the format of the chat completions API: this is all the model sees."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _tool("search_corpus",
          "Search the team's document corpus for relevant passages. Use this whenever the user asks about the "
          "documents, their rules, facts, figures, dates or definitions. Returns the best matching passages, each "
          "with its [Source: file p.N], or 'No matching passages found' when nothing is relevant. Search one topic "
          "at a time; if nothing useful comes back, rephrase with other keywords.",
          {"query": {"type": "string", "description": "What to search for, phrased as a short search query."}},
          ["query"]),
    _tool("calculator",
          "Evaluate an arithmetic expression and return the exact result. Use it for ANY calculation "
          "(percentages, sums, differences, averages, unit conversions), never compute in your head.",
          {"expression": {"type": "string", "description": "Digits and operators only, e.g. '10000 * 0.15' or "
                                                           "'(1200 + 850) / 2'. No words, units or % signs."}},
          ["expression"]),
    _tool("quote_exact",
          "Return the exact full text of one page of one document. Use it after search_corpus pointed to a page, "
          "to read the whole page, quote it word for word, or verify a claim.",
          {"source": {"type": "string", "description": "Document file name, e.g. 'guide.pdf'."},
           "page": {"type": "integer", "description": "Page number (1 for web pages and text files)."}},
          ["source", "page"]),
    _tool("get_article",
          "Return the exact, complete text of one EU AI Act Article by number (1 to 113), with its title and pages, "
          "or only one numbered paragraph (article=99, paragraph=3 for Article 99(3)). Use it whenever the question "
          "names an Article, instead of relying on search excerpts that may be cut off. EU AI Act only: never use "
          "it for the NIST AI RMF.",
          {"article": {"type": "integer", "description": "Article number, 1 to 113."},
           "paragraph": {"type": "integer", "description": "Optional paragraph number, e.g. 3 for 99(3)."}},
          ["article"]),
    _tool("date_tool",
          "Date arithmetic, which the calculator cannot do. operation='today' returns today's date. "
          "operation='days_between' returns the number of days from start to end, in a sentence that says whether "
          "the end is in the past or the future. Dates as YYYY-MM-DD, '2 August 2026' or 'today'. Use it for any "
          "'how long until / since / between' question; never count days yourself. Only use dates you have read "
          "in a tool result, never dates from memory.",
          {"operation": {"type": "string", "enum": ["today", "days_between"]},
           "start": {"type": "string", "description": "Start date (days_between)."},
           "end": {"type": "string", "description": "End date (days_between)."}},
          ["operation"]),
    _tool("send_message",
          "Post a message to the team's Slack/Teams channel. This is irreversible, so a person must approve it "
          "before it is sent. Only use it when the user explicitly asks to send, post or share something to the "
          "channel. Pass the complete text to post, including citations.",
          {"text": {"type": "string", "description": "The full message to post."},
           "channel": {"type": "string", "description": "Channel ID. Leave empty to use the team's default channel."}},
          ["text"]),
]
TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]


def search_corpus(query: str, seen_ids: set | None = None, k: int | None = None) -> tuple[str, list[dict]]:
    """Passages for the agent, as (text for the model, new results). Passages already returned
    in this run (seen_ids) are only mentioned by reference, not sent again."""
    query = (query or "").strip()
    if not query:
        return "Error: empty query. Pass what to search for.", []
    seen_ids = seen_ids if seen_ids is not None else set()
    results = retrieve(query, k=k or config.AGENT_TOP_K)
    if not is_confident(results, query):
        best = best_score(results)
        best_text = "none" if best is None else f"{best:.2f}"
        return (f"No matching passages found (best {score_name(results)} {best_text}, threshold "
                f"{threshold(results)}). Try different keywords, or tell the user the documents don't cover this."), []

    new = [r for r in results if r["id"] not in seen_ids]
    repeated = [r for r in results if r["id"] in seen_ids]
    seen_ids.update(r["id"] for r in new)
    lines = [f"[Source: {r['source']} {page_label(r)}] ({_score_text(r)})\n{r['text']}" for r in new]
    if repeated:
        lines.append(f"(Also matched passages already shown earlier: "
                     f"{', '.join(r['source'] + ' ' + page_label(r) for r in repeated)}.)")
    if not new:
        lines.insert(0, "No new passages: everything this query matches was already returned above. "
                        "Use a different query, quote_exact to read a full page, or answer.")
    note = unknown_words_note(query)
    if note:
        lines.append(note)
    return "\n\n".join(lines), new


def _score_text(result: dict) -> str:
    if result.get("rerank_score") is not None:
        return f"reranker score {result['rerank_score']:.2f}"
    return f"similarity {result['score']:.2f}"


def calculator(expression: str) -> str:
    """Safe arithmetic. simpleeval, never eval(): the expression comes from the model, and maybe from a document."""
    expr = (expression or "").strip()
    if not expr:
        return "Error: empty expression."
    if re.search(r"\d,\d", expr):
        return ("Error: commas are not allowed in numbers. Use '.' for decimals and no thousands "
                "separators, e.g. '10000 * 0.15'.")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        result = simple_eval(expr)
    except Exception as e:
        return (f"Error: could not evaluate {expression!r} ({type(e).__name__}: {e}). "
                "Use digits and operators only, e.g. '10000 * 0.15'.")
    if isinstance(result, float):
        result = round(result, 10)
        result = int(result) if result.is_integer() else result
    if not isinstance(result, (int, float)):
        return f"Error: {expression!r} is not an arithmetic expression."
    return str(result)


def _cap(text: str, hint: str) -> str:
    if len(text) <= config.QUOTE_MAX_CHARS:
        return text
    return text[:config.QUOTE_MAX_CHARS] + f"\n[... truncated at {config.QUOTE_MAX_CHARS} characters{hint}]"


def _match_source(name: str, sources: list[str]) -> str | None:
    """Exact name, then any case, then without the extension. None if unknown or ambiguous."""
    if name in sources:
        return name
    lowered = {s.lower(): s for s in sources}
    if name.lower() in lowered:
        return lowered[name.lower()]
    stem = name.lower().rsplit(".", 1)[0]
    candidates = [s for s in sources if s.lower().rsplit(".", 1)[0] == stem]
    return candidates[0] if len(candidates) == 1 else None


def quote_exact(source: str, page) -> str:
    """The exact text of one page, from corpus.json."""
    index = get_index()
    if not index.pages:
        return "Error: data/corpus.json was not found, so pages can't be quoted."
    sources = index.sources
    matched = _match_source((source or "").strip(), list(sources))
    if matched is None:
        close = difflib.get_close_matches(source or "", list(sources), n=5, cutoff=0.3)
        return f"Error: unknown document {source!r}. Known documents include: {', '.join(close or list(sources)[:10])}."
    try:
        page = int(page)
    except (TypeError, ValueError):
        return f"Error: page must be an integer, got {page!r}."
    if (matched, page) not in index.pages:
        pages = sources[matched]
        return f"Error: {matched} has no page {page}. Valid pages: {pages[0]} to {pages[-1]}."
    text = _cap(index.pages[(matched, page)], "")
    return f"[Source: {matched} p.{page}] exact text:\n{text or '(this page has no extractable text)'}"


_ARTICLE_HEADING = re.compile(r"Article\s+(\d+)")
_ARTICLE_END = re.compile(r"(CHAPTER\s+[IVXLC]+|SECTION\s+\d+|ANNEX\s+[IVXLC]+|Done at\b)")
LAST_ARTICLE = 113


def _eu_act_source() -> str | None:
    return next((s for s in get_index().sources if "eu_ai_act" in s.lower()), None)


@lru_cache(maxsize=1)
def load_articles() -> dict:
    """Article number -> {"title", "pages", "body": [(page, paragraph text), ...]}, built once.
    Headings are paragraphs that are exactly "Article N", in increasing order, so references
    like "Article 49" inside the Annexes are skipped."""
    source = _eu_act_source()
    if source is None:
        return {}
    blocks = [(page, block.strip())
              for (s, page), text in sorted(get_index().pages.items()) if s == source
              for block in text.split("\n\n") if block.strip()]
    starts, expected = [], 1
    for i, (_, block) in enumerate(blocks):
        match = _ARTICLE_HEADING.fullmatch(block)
        if match and int(match.group(1)) == expected:
            starts.append((expected, i))
            expected += 1
            if expected > LAST_ARTICLE:
                break
    articles = {}
    for n, (number, start) in enumerate(starts):
        end = starts[n + 1][1] if n + 1 < len(starts) else len(blocks)
        body = []
        for page, block in blocks[start + 2:end]:  # after the heading and the title
            if _ARTICLE_END.match(block):
                break
            body.append((page, block))
        title = blocks[start + 1][1].strip("`' ") if start + 1 < len(blocks) else ""
        articles[number] = {"title": title, "pages": sorted({blocks[start][0]} | {p for p, _ in body}), "body": body}
    return articles


def _paragraph(body: list, number: int) -> list | None:
    """The blocks of paragraph 'N.' or, when there is none, of point '(N)'."""
    for style in (r"{n}\.\s", r"\({n}\)\s"):
        first, following = re.compile(style.format(n=number)), re.compile(style.format(n=number + 1))
        start = next((i for i, (_, b) in enumerate(body) if first.match(b)), None)
        if start is not None:
            end = next((i for i in range(start + 1, len(body)) if following.match(body[i][1])), len(body))
            return body[start:end]
    return None


def get_article(article, paragraph=None) -> str:
    """Our own tool: the exact text of an EU AI Act Article, or of one of its paragraphs."""
    try:
        article = int(article)
        paragraph = int(paragraph) if paragraph not in (None, "", 0) else None
    except (TypeError, ValueError):
        return "Error: article and paragraph must be integers, e.g. article=99, paragraph=3."
    if not 1 <= article <= LAST_ARTICLE:
        return f"Error: the EU AI Act has Articles 1 to {LAST_ARTICLE}; there is no Article {article}."
    articles = load_articles()
    if article not in articles:
        return f"Error: Article {article} was not found in the corpus. Use search_corpus instead."
    entry, body = articles[article], articles[article]["body"]
    if paragraph is not None:
        body = _paragraph(body, paragraph)
        if body is None:
            return (f"Error: Article {article} has no paragraph {paragraph}. "
                    "Call get_article again without a paragraph to read the whole Article.")
    pages = sorted({page for page, _ in body}) or entry["pages"]
    text = _cap("\n\n".join(block for _, block in body), ": ask for one paragraph, or use quote_exact on a later page")
    reference = f"Article {article}" + (f"({paragraph})" if paragraph else "")
    return f"[Source: {_eu_act_source()} {page_label({'pages': pages})}] {reference} · {entry['title']}, exact text:\n{text}"


_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y")


def parse_date(value) -> date | None:
    """2026-08-02, '2 August 2026', 'August 2, 2026' or 'today'. 02/08/2026 is refused on
    purpose: 2 August or February 8?"""
    if not isinstance(value, str):
        return None
    text = " ".join(value.replace(",", " ").split())
    if text.lower() == "today":
        return date.today()
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    return None


def _readable(day: date) -> str:
    return f"{day.day} {day:%B %Y}"


def date_tool(operation: str, start=None, end=None) -> str:
    """Our own tool: today's date, or the days between two dates, written as a sentence so the
    model cannot misread the direction (it once read '+59 days' as '59 days away')."""
    if operation == "today":
        today = date.today()
        return f"Today is {today:%A} {_readable(today)} ({today.isoformat()})."
    if operation != "days_between":
        return "Error: operation must be 'today' or 'days_between'."
    first, second = parse_date(start), parse_date(end)
    for name, value, parsed in (("start", start, first), ("end", end, second)):
        if parsed is None:
            return f"Error: could not read the {name} date {value!r}. Use YYYY-MM-DD, '2 August 2026' or 'today'."
    days = (second - first).days
    today = date.today()
    if second == today and days > 0:
        sentence = f"{_readable(first)} was {days} days ago (in the past)."
    elif first == today and days > 0:
        sentence = f"{_readable(second)} is in {days} days (in the future)."
    elif days == 0:
        sentence = "Both dates are the same day (0 days)."
    else:
        sentence = (f"From {_readable(first)} to {_readable(second)} there are {abs(days)} days"
                    + (" (the end date is before the start date)." if days < 0 else "."))
    return f"{sentence} [days_between = {days}; {abs(days) / 7:.1f} weeks, {abs(days) / 365.25:.2f} years]"
