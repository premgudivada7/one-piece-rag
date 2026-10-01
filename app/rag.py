import re
import time

from app import config
from app.chunking import page_label
from app.index import get_index
from app.llm import ask
from app.prompts import CONDENSE_PROMPT, EXPANSION_PROMPT, REFUSAL, context_block, rag_system_prompt
from app.retrieval import best_score, is_confident, merge_results, retrieve, unknown_words_note

# Refusals in English or French (the model translates REFUSAL for French questions).
REFUSAL_PATTERN = re.compile(
    r"(don't|do not|doesn't|does not) (have|contain) (enough )?information"
    r"|not (covered|mentioned|specified|provided) in the (documents|context|passages)"
    r"|only (answer|help with) questions about the documents"
    r"|je n'ai pas (d'|assez d')?information|ne (contiennent|contient) pas",
    re.IGNORECASE)
# A file name followed by one or more pages, in any style the model uses:
# [guide.pdf p.3]  (guide.pdf, p.3-4)  guide.pdf p.47-48, p.26  guide.pdf pp. 3–4
_PAGE = r"pp?\.?\s?\d+(?:\s?[-–]\s?\d+)?"
CITATION_PATTERN = re.compile(
    rf"([\w\-./]+\.(?:pdf|html?|txt|md))[\s,:]*((?:{_PAGE})(?:\s*(?:,|;|and)\s*{_PAGE})*)", re.IGNORECASE)
PAGE_NUMBERS = re.compile(r"(\d+)(?:\s?[-–]\s?(\d+))?")
FOLLOW_UP_LINE = re.compile(r"\n*[ *_]*follow-up questions?[ *_]*:[ *_]*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_REFERRING = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|there|above|previous|same|"
                        r"also|else|what about|how about|and)\b", re.IGNORECASE)


def _citations(answer: str) -> list[tuple[str, int, int]]:
    """(source, first page, last page) for every citation in the answer, in order."""
    return [(source, int(first), int(last or first))
            for source, pages in CITATION_PATTERN.findall(answer or "")
            for first, last in PAGE_NUMBERS.findall(pages)]


def is_refusal(answer: str) -> bool:
    """True when the answer declines. An answer with a citation answered at least part of the question."""
    return bool(answer) and bool(REFUSAL_PATTERN.search(answer)) and not CITATION_PATTERN.search(answer)


def cited_sources(answer: str) -> set[tuple[str, int]]:
    return {(source, page) for source, first, last in _citations(answer) for page in range(first, last + 1)}


def citation_labels(answer: str) -> list[str]:
    """Citations as written in the answer, e.g. ['guide.pdf p.3', 'faq.html p.1'], without repeats."""
    labels = [f"{source} p.{first}" + (f"-{last}" if last != first else "") for source, first, last in _citations(answer)]
    return list(dict.fromkeys(labels))


def is_cited(chunk: dict, answer: str) -> bool:
    """Does the answer cite this chunk, by its label ('guide.pdf p.4-5') or one of its pages?"""
    labels = {label.lower() for label in citation_labels(answer)}
    pages = [page_label(chunk)] + [f"p.{p}" for p in chunk.get("pages") or [chunk["page"]]]
    return any(f"{chunk['source']} {p}".lower() in labels for p in pages)


def format_passages(results: list[dict]) -> str:
    return "\n\n".join(f"[Source: {r['source']} {page_label(r)}]\n{r['text']}" for r in results)


def recent_history(history: list[dict] | None) -> list[dict]:
    """The last HISTORY_TURNS exchanges, with only the start of long answers (keeps calls cheap)."""
    recent = (history or [])[-2 * config.HISTORY_TURNS:]
    return [{"role": m["role"], "content": m["content"] if m["role"] == "user"
             else m["content"][:config.HISTORY_ANSWER_CHARS]} for m in recent]


def looks_like_follow_up(question: str) -> bool:
    """Short questions and ones that refer back ("and is it paid?") get rewritten; a standalone
    question skips the extra call."""
    return len(question.split()) <= 5 or bool(_REFERRING.search(question))


def rewrite_question(question: str, history: list[dict]) -> tuple[str, float]:
    """Turn a follow-up ("and for knockout rounds?") into a standalone search query."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent_history(history))
    text, cost = ask([{"role": "system", "content": CONDENSE_PROMPT},
                      {"role": "user", "content": f"Conversation:\n{transcript}\n\nLast message: {question}"}],
                     operation="rag_condense", max_tokens=120)
    return text or question, cost


def expand_query(question: str) -> tuple[list[str], float]:
    """Query expansion: a few rephrasings of the question in the documents' own words."""
    documents = ", ".join(get_index().source_names)
    text, cost = ask([{"role": "system", "content": EXPANSION_PROMPT.format(n=config.EXPANSION_VARIANTS,
                                                                            documents=documents)},
                      {"role": "user", "content": question}], operation="rag_expand", max_tokens=150)
    lines = [line.strip(" -*•\t\"") for line in text.splitlines()]
    variants = [line for line in lines if len(line) > 3 and line.lower() != question.lower()]
    return variants[:config.EXPANSION_VARIANTS], cost


def split_follow_ups(text: str) -> tuple[str, list[str]]:
    """(answer without the "Follow-up questions:" line, up to 3 suggested questions)."""
    match = FOLLOW_UP_LINE.search(text or "")
    if not match:
        return (text or "").strip(), []
    questions = [q.strip(" -*•\n") for q in re.split(r"\s*\|\s*|\n+", match.group(1))]
    return text[:match.start()].strip(), [q for q in questions if len(q) > 5][:3]


def search(question: str, history: list[dict] | None = None, k: int | None = None, mode: str | None = None,
           rerank: bool | None = None, rewrite: bool | None = None, expand: bool | None = None) -> dict:
    """Everything before the answer: rewrite a follow-up, retrieve and rerank, check confidence,
    and optionally expand the query. Used by the app and by the retrieval-only eval."""
    rewrite = config.USE_QUERY_REWRITE if rewrite is None else rewrite
    expand = config.QUERY_EXPANSION if expand is None else expand
    start, timings, cost, calls = time.time(), {}, 0.0, 0
    search_query = question
    if rewrite and history and looks_like_follow_up(question):
        search_query, cost = rewrite_question(question, history)
        calls += 1
        timings["condense"] = round(time.time() - start, 2)
    t0 = time.time()
    results = retrieve(search_query, k=k, mode=mode, rerank=rerank)
    timings["retrieval"] = round(time.time() - t0, 2)
    confident = is_confident(results, search_query)
    variants = []
    if expand and confident:  # only spend the extra call when we are going to answer
        t0 = time.time()
        variants, expand_cost = expand_query(search_query)
        cost, calls = cost + expand_cost, calls + 1
        lists = [results] + [retrieve(v, k=k, mode=mode, rerank=rerank) for v in variants]
        results = merge_results(lists, k or config.TOP_K)
        timings["expansion"] = round(time.time() - t0, 2)
    return {"search_query": search_query, "results": results, "confident": confident, "top_score": best_score(results),
            "expanded_queries": variants, "cost": cost, "model_calls": calls, "timings": timings}


def generate(question: str, search_query: str, results: list[dict], history: list[dict] | None,
             with_follow_ups: bool) -> tuple[str, list[str], float]:
    """The model answers the original question from the passages, with the recent conversation."""
    note = unknown_words_note(search_query)
    user = f"{context_block(format_passages(results))}\n\nQuestion: {question}" + (f"\n\n{note}" if note else "")
    text, cost = ask([{"role": "system", "content": rag_system_prompt(with_follow_ups)}, *recent_history(history),
                      {"role": "user", "content": user}], operation="rag_answer", max_tokens=config.RAG_MAX_TOKENS)
    answer, follow_ups = split_follow_ups(text)
    return answer, follow_ups, cost


def answer_question(question: str, history: list[dict] | None = None, k: int | None = None,
                    mode: str | None = None, with_follow_ups: bool = False) -> dict:
    """The full RAG pipeline for one question. `history` is a list of {"role", "content"} messages."""
    start = time.time()
    found = search(question, history, k=k, mode=mode)
    answer, follow_ups, timings = REFUSAL, [], found["timings"]
    if found["confident"]:
        t0 = time.time()
        answer, follow_ups, answer_cost = generate(question, found["search_query"], found["results"], history,
                                                   with_follow_ups)
        timings["generation"] = round(time.time() - t0, 2)
        found["cost"] += answer_cost
        found["model_calls"] += 1
    refused = is_refusal(answer)
    timings["total"] = round(time.time() - start, 2)
    return {**found, "question": question, "answer": answer, "below_threshold": not found["confident"],
            "refused": refused, "cited": sorted(cited_sources(answer)), "follow_ups": [] if refused else follow_ups,
            "latency_s": timings["total"]}
