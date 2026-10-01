import html
import math
import re
from urllib.parse import quote

from app import config
from app.chunking import page_label
from app.rag import is_cited
from app.retrieval import score_name, threshold

STOPWORDS = set("""a an and are as at be by can do does for from has have how in is it its of on or
that the their there this to was what when where which who why will with under about into than
then these those shall should would could may must not no yes any all each our your they them""".split())

CSS = """
.panel {max-height: 720px; overflow-y: auto;}
.approval {border: 2px solid #f59e0b; border-radius: 10px; padding: 8px;}
.src-summary {font-weight: 600; margin: 4px 0 12px;}
.src-card {border: 1px solid var(--border-color-primary); border-left: 4px solid var(--border-color-primary);
  border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; background: var(--background-fill-primary);}
.src-card.cited {border-left-color: #16a34a;}
.src-head {display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 6px;}
.src-num {background: var(--color-accent); color: white; border-radius: 6px; padding: 1px 8px;
  font-weight: 700; font-size: 13px;}
.tag {font-size: 11.5px; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued);}
.tag.cited {background: rgba(22, 163, 74, 0.14); border-color: rgba(22, 163, 74, 0.5); color: var(--body-text-color);}
.src-title {font-weight: 700; font-size: 15px; margin: 2px 0;}
.src-open {display: inline-block; font-size: 13px; font-weight: 600; margin: 2px 0 4px; color: var(--color-accent);
  text-decoration: none;}
.src-open:hover {text-decoration: underline;}
.src-metrics {font-size: 12.5px; color: var(--body-text-color-subdued);}
.src-bar {height: 6px; border-radius: 3px; background: var(--border-color-primary); margin: 6px 0 8px;}
.src-bar > div {height: 6px; border-radius: 3px; background: var(--color-accent);}
.src-text {font-size: 13.5px; line-height: 1.5;}
.src-text mark, .src-full mark {background: rgba(250, 204, 21, 0.45); color: inherit; padding: 0 1px;}
.src-card details summary {cursor: pointer; font-size: 12.5px; color: var(--body-text-color-subdued); margin-top: 6px;}
.src-full {font-size: 13px; line-height: 1.5; white-space: pre-wrap; margin-top: 6px;}
.status-line {font-size: 14px; color: var(--body-text-color-subdued);}
"""


def query_terms(question: str) -> list[str]:
    """Words of the question worth highlighting (no stopwords, 3+ letters)."""
    words = re.findall(r"[\w\-.]+", question.lower())
    terms = {w.strip(".-") for w in words if len(w.strip(".-")) >= 3 and w.strip(".-") not in STOPWORDS}
    return sorted(terms, key=len, reverse=True)


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escape the text and wrap each question word in <mark>."""
    escaped = html.escape(text)
    if not terms:
        return escaped
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(html.escape(t)) for t in terms) + r")(?!\w)", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def document_url(source: str, page: int | None = None) -> str | None:
    """Link that opens the original file from data/sources/, at the given page for a PDF."""
    sources_dir = config.SOURCES_DIR.resolve()
    path = (sources_dir / source).resolve()
    if path.parent != sources_dir or not path.is_file():
        return None
    url = "/gradio_api/file=" + quote(str(path))
    if page and path.suffix.lower() == ".pdf":
        url += f"#page={page}"
    return url


def document_link_markdown(label: str) -> str:
    """'guide.pdf p.4-5' -> a Markdown link opening guide.pdf at page 4 (or the label as is)."""
    match = re.match(r"(.+?)\s+pp?\.\s?(\d+)", label)
    url = document_url(match.group(1), int(match.group(2))) if match else None
    return f"[{label}]({url})" if url else label


def source_title(r: dict) -> str:
    parts = [r["source"]]
    if r.get("section"):
        parts.append(r["section"])
    parts.append(page_label(r))
    return " · ".join(parts)


def source_cards(results: list[dict], answer: str, question: str, mode: str = "", note: str = "",
                 refused: bool = False) -> str:
    """The Sources panel: a summary line and one card per passage. After a refusal the passages
    are folded away, so they can't be mistaken for an answer."""
    if not results:
        return "<div class='src-summary'>No passages yet: ask a question.</div>"
    terms = query_terms(question)
    cited = [r for r in results if is_cited(r, answer)]
    cited_pages = {(r["source"], p) for r in cited for p in r.get("pages") or [r["page"]]}
    summary = (f"{len(cited)} cited passage(s) on {len(cited_pages)} page(s) · "
               f"{len(results)} passages retrieved ({mode or config.RETRIEVAL_MODE} search)")
    if refused:
        summary = "No answer: the documents don't cover this question."
    head = [f"<div class='src-summary'>{html.escape(summary)}</div>"]
    if note:
        head.append(f"<div class='src-metrics' style='margin-bottom:10px'>{html.escape(note)}</div>")
    cards = []
    for i, r in enumerate(results, start=1):
        is_c = r in cited
        tags = ["<span class='tag cited'>cited in answer</span>" if is_c else
                "<span class='tag'>retrieved, not cited</span>"]
        if r.get("bm25_rank"):
            tags.append("<span class='tag'>keyword</span>")
        if r.get("vector_rank"):
            tags.append("<span class='tag'>vector</span>")
        if r.get("reference"):
            tags.append("<span class='tag'>exact reference</span>")
        if r.get("injected"):
            tags.append("<span class='tag'>⚠️ injection test passage</span>")
        metrics = []
        if r.get("rerank_score") is not None:
            metrics.append(f"reranker {r['rerank_score']:.2f} (search rank {r.get('search_rank')})")
        if r.get("score") is not None:
            metrics.append(f"similarity {r['score']:.3f}")
        if r.get("bm25"):
            metrics.append(f"BM25 {r['bm25']:.1f}")
        if r.get("rrf"):
            metrics.append(f"RRF {r['rrf']:.4f}")
        width = bar_width(r)
        text = r["text"]
        preview = text[:420] + ("…" if len(text) > 420 else "")
        first_page = (r.get("pages") or [r["page"]])[0]
        url = document_url(r["source"], first_page)
        where = f" at page {first_page}" if r["source"].lower().endswith(".pdf") else ""
        open_link = (f"<a class='src-open' href='{html.escape(url)}' target='_blank' rel='noopener'>"
                     f"📄 Open {html.escape(r['source'])}{where}</a>" if url else "")
        cards.append(
            f"<div class='src-card{' cited' if is_c else ''}'>"
            f"<div class='src-head'><span class='src-num'>{i}</span>{''.join(tags)}</div>"
            f"<div class='src-title'>{html.escape(source_title(r))}</div>"
            f"{open_link}"
            f"<div class='src-metrics'>{' · '.join(metrics)}</div>"
            f"<div class='src-bar'><div style='width:{width:.0f}%'></div></div>"
            f"<div class='src-text'>{highlight(preview.replace(chr(10), ' '), terms)}</div>"
            f"<details><summary>show full passage</summary>"
            f"<div class='src-full'>{highlight(text, terms)}</div></details></div>")
    if refused:
        return ("".join(head) + f"<details class='src-rejected'><summary>Show the {len(results)} rejected passages "
                "(what the search found, none of it answers the question)</summary>" + "".join(cards) + "</details>")
    return "".join(head + cards)


def bar_width(r: dict) -> float:
    """Relevance bar in %: the reranker score through a sigmoid, or the cosine similarity."""
    if r.get("rerank_score") is not None:
        return 100 / (1 + math.exp(-r["rerank_score"]))
    return max(0.0, min(1.0, r.get("score") or 0.0)) * 100


def _via(r: dict) -> str:
    via = [name for name, on in (("keyword", r.get("bm25_rank")), ("vector", r.get("vector_rank")),
                                 ("reference", r.get("reference"))) if on]
    return "+".join(via) or "fusion"


def embedder_name() -> str:
    if config.EMBED_PROVIDER == "local":
        return f"{config.LOCAL_EMBED_MODEL.rstrip('/').split('/')[-1]} (sentence-transformers, local)"
    return f"{config.EMBED_MODEL} (OpenAI API)"


def chat_tool_name() -> str:
    api = "xAI API" if config.LLM_PROVIDER == "xai" else "OpenAI API"
    return f"{config.CHAT_MODEL} ({api}, openai library)"


def reranker_name() -> str:
    return f"{config.RERANK_MODEL.rstrip('/').split('/')[-1]} (sentence-transformers cross-encoder, local)"


def steps_markdown(result: dict) -> tuple[str, str]:
    """(title, body) of the collapsible "Steps & tools used" message above the answer."""
    t = result["timings"]
    results = result["results"]
    steps, tools = [], []
    if "condense" in t:
        tools.append("Follow-up rewrite")
        steps.append(f"**🔁 Follow-up rewritten** ({t['condense']}s) · *tool: {chat_tool_name()}*  \n"
                     f"    as: “{result['search_query']}”")
    tools.append("Hybrid search (BM25 + vectors + exact references)")
    pages = {(r["source"], p) for r in results for p in r.get("pages") or [r["page"]]}
    reranked = any(r.get("rerank_score") is not None for r in results)
    lines = [f"**🔎 Search** ({t['retrieval']}s): {len(results)} passages from {len(pages)} pages (top-{config.TOP_K})  \n"
             f"    *tools: {embedder_name()} embeds the question · FAISS finds similar passages · "
             f"rank_bm25 finds keywords · exact-reference matcher · Reciprocal Rank Fusion merges them"
             + (f" · {reranker_name()} reranks the best {config.RERANK_CANDIDATES}*" if reranked else "*")]
    if not result["below_threshold"]:  # after a refusal the passages are not listed: they are not an answer
        for i, r in enumerate(results, start=1):
            score = f"rerank {r['rerank_score']:.2f}" if r.get("rerank_score") is not None else f"sim {r['score']:.2f}"
            lines.append(f"    - [{i}] {source_title(r)} · {score} · via {_via(r)}")
    steps.append("\n".join(lines))
    if result.get("expanded_queries"):
        tools.append("Query expansion")
        queries = " · ".join(f"“{q}”" for q in result["expanded_queries"])
        steps.append(f"**🔀 Query expansion** ({t.get('expansion', 0)}s): also searched for {queries}, "
                     f"merged with Reciprocal Rank Fusion · *tool: {chat_tool_name()}*")
    tools.append("Confidence check")
    best, name, limit = result["top_score"], score_name(results), threshold(results)
    best_text = "none" if best is None else f"{best:.2f}"
    if result["below_threshold"]:
        steps.append(f"**🛡️ Confidence check**: best {name} {best_text} < {limit} → refused, the model was not called")
    else:
        steps.append(f"**🛡️ Confidence check**: best {name} {best_text} ≥ {limit} → answer")
        tools.append(f"Answer written by {config.CHAT_MODEL}")
        steps.append(f"**✍️ Answer written by {config.CHAT_MODEL}** ({t.get('generation', 0)}s), "
                     f"{len(result['cited'])} cited page(s) · *tool: {chat_tool_name()}*")
        if result["follow_ups"]:
            tools.append("Follow-up suggestions")
            steps.append(f"**💡 Follow-up suggestions**: {len(result['follow_ups'])} questions · "
                         f"*tool: same {config.CHAT_MODEL} call*")
    body = f"**Tools used:** {' → '.join(tools)}\n\n" + "\n".join(f"{n}. {step}" for n, step in enumerate(steps, start=1))
    return f"🛠️ Steps & tools used ({len(steps)} steps) · {t['total']}s", body


def how_it_works_markdown(n_docs: int | None = None, n_chunks: int | None = None) -> str:
    """The "How it works" tab: every tool we use and the live settings."""
    corpus = f"{n_docs} documents, {n_chunks} chunks" if n_docs else "index not loaded"
    rerank = f"cross-encoder `{config.RERANK_MODEL}`" if config.USE_RERANK else "off"
    limit = f"{config.RERANK_MIN_SCORE} (cross-encoder score)" if config.USE_RERANK else f"{config.MIN_SCORE} (cosine)"
    return f"""## How it works

**We find the right pages first, by meaning and by exact words, rerank them with a cross-encoder, then
{config.CHAT_MODEL} answers only from those pages and cites them.**

### Tools used

| Step | Tool | What it does |
|---|---|---|
| Read the PDFs | **PyMuPDF** | Extracts the text page by page into `corpus.json` |
| Chunking | `app/chunking.py` | ~{config.CHUNK_SIZE}-character passages cut on line and sentence boundaries, {config.CHUNK_OVERLAP} characters of overlap |
| Embeddings | {embedder_name()} | Turns passages and questions into vectors. Runs on this machine, no key |
| Vector search | **FAISS** (`IndexFlatIP`) | Passages closest in meaning to the question |
| Keyword search | **rank_bm25** | Passages containing the question's exact words |
| Exact references | `app/index.py` tokenizer | Matches "Article 9" or "GOVERN 1.1" exactly |
| Merge | Reciprocal Rank Fusion | Combines the three lists |
| Rerank | {reranker_name()} | Reads the question and each of the best {config.RERANK_CANDIDATES} passages together and keeps the top {config.TOP_K} |
| Confidence check | reranker score | Refuses without calling the model when the best passage scores below the threshold |
| Answer | `{config.CHAT_MODEL}` through the **openai** library ({'xAI API' if config.LLM_PROVIDER == 'xai' else 'OpenAI API'}) | Answers from the passages only, with a `[file p.N]` citation per fact |
| Web page | **Gradio** | This interface |

**Agent tab tools:** `search_corpus` (the search above, top {config.AGENT_TOP_K}) · `calculator` (**simpleeval**) ·
`quote_exact` (one full page) · `get_article` (an EU AI Act Article or paragraph, our tool) · `date_tool` (days
between dates, our tool) · `send_message` (Slack webhook, only after a person approves).

No LangChain, no LlamaIndex, no agent framework: the pipeline and the agent loop are our own Python code.

### Current settings

| Setting | Value |
|---|---|
| Corpus | {corpus} |
| Retrieval | {config.RETRIEVAL_MODE}, top-{config.TOP_K} |
| Reranking | {rerank} |
| Refusal threshold | {limit} |
| Follow-up rewrite | {'on' if config.USE_QUERY_REWRITE else 'off'}, memory of the last {config.HISTORY_TURNS} exchanges |
| Chat model | `{config.CHAT_MODEL}` ({config.LLM_PROVIDER}) |
| Agent step cap | {config.AGENT_MAX_STEPS} model calls |
| Prompt injection test | {'⚠️ ON' if config.INJECTION_TEST else 'off'} |
"""


def status_line(result: dict) -> str:
    t = result["timings"]
    best = result["top_score"]
    parts = ["Done", f"retrieval {t['retrieval']}s"]
    if "generation" in t:
        parts.append(f"generation {t['generation']}s")
    parts += [f"total {t['total']}s", f"best {score_name(result['results'])} {'none' if best is None else f'{best:.2f}'}",
              f"cost ${result['cost']:.5f}"]
    return "<div class='status-line'>" + " · ".join(parts) + "</div>"
