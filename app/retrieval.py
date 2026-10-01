import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from app import config
from app.index import embedding_text, get_index, tokenize
from app.llm import embed_query

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """The cross-encoder, loaded once (downloads about 90 MB the first time)."""
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"loading reranker {config.RERANK_MODEL} ...")
            _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def rerank_scores(query: str, chunks: list[dict]) -> list[float]:
    """Cross-encoder relevance of each chunk to the question. Raw scores: higher is more
    relevant, roughly -11 (unrelated) to +11 (answers it)."""
    import torch
    pairs = [(query, embedding_text(c)) for c in chunks]
    scores = get_reranker().predict(pairs, activation_fn=torch.nn.Identity(), show_progress_bar=False)
    return [float(s) for s in scores]


def _vector_ranking(index, query_vec, n):
    _, ids = index.faiss.search(query_vec.reshape(1, -1), n)
    return [int(i) for i in ids[0] if i != -1]


def _bm25_ranking(scores, n):
    order = np.argsort(scores)[::-1][:n]
    return [int(i) for i in order if scores[i] > 0]


def reference_tokens(query: str) -> set[str]:
    """Joined references in the question, like 'article_9' or 'govern_1.1'."""
    return {t for t in tokenize(query) if "_" in t}


def _reference_ranking(index, query, bm25_ids):
    """Chunks containing a reference the question names ('Article 9', 'GOVERN 1.1'), most
    references first. Embeddings blur exact references: the GOVERN 1.1 row ranks 30th by meaning."""
    refs = reference_tokens(query)
    if not refs:
        return []
    matches = [i for i in bm25_ids if refs & index.token_sets[i]]
    return sorted(matches, key=lambda i: -len(refs & index.token_sets[i]))


def _fuse(rankings: list[list[int]]) -> tuple[list[int], dict]:
    """Reciprocal Rank Fusion: each list gives 1 / (60 + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
    return sorted(fused, key=fused.get, reverse=True), fused


def mentioned_sources(query: str, sources: list[str]) -> list[str]:
    """Documents the question names, from the words only their file name has
    ('nist', 'rmf' for nist_ai_rmf_1_0.pdf; 'ai' is in both names, so it names neither)."""
    words = {s: set(re.findall(r"[a-z]{2,}", Path(s).stem.lower())) for s in sources}
    counts = Counter(w for ws in words.values() for w in ws)
    query_words = set(re.findall(r"[a-z]{2,}", query.lower()))
    return [s for s, ws in words.items() if any(counts[w] == 1 and w in query_words for w in ws)]


def _take(ranked: list[int], chunks: list[dict], named: list[str], k: int) -> list[int]:
    """The best k chunks; when the question names two documents, take them in turns so one
    document cannot fill every slot."""
    if len(named) < 2:
        return ranked[:k]
    per_source = {s: [i for i in ranked if chunks[i]["source"] == s] for s in named}
    order = sorted(named, key=lambda s: ranked.index(per_source[s][0]) if per_source[s] else len(ranked))
    chosen = []
    while len(chosen) < k and any(per_source.values()):
        for s in order:
            if per_source[s] and len(chosen) < k:
                chosen.append(per_source[s].pop(0))
    return chosen


def retrieve(query: str, k: int = None, mode: str = None, rerank: bool = None) -> list[dict]:
    """Top-k chunks for the question, best first. Each result is a copy of the chunk plus:
    score (cosine), rerank_score (cross-encoder, None without reranking), vector_rank, bm25_rank,
    search_rank (rank before reranking), bm25, reference, rrf."""
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    rerank = config.USE_RERANK if rerank is None else rerank
    index = get_index()
    query_vec = embed_query(query)
    if query_vec.shape[-1] != index.faiss.d:
        raise ValueError(f"The index holds {index.faiss.d}-dimension vectors but {config.EMBED_MODEL} makes "
                         f"{query_vec.shape[-1]}: rebuild it with python scripts/build_index.py --force")
    named = mentioned_sources(query, index.source_names)
    n = index.faiss.ntotal if len(named) >= 2 else max(config.CANDIDATES, k, config.RERANK_CANDIDATES)

    vector_ids = _vector_ranking(index, query_vec, n) if mode in ("hybrid", "vector") else []
    bm25_scores = index.bm25.get_scores(tokenize(query))
    bm25_ids = _bm25_ranking(bm25_scores, n) if mode in ("hybrid", "bm25") else []
    reference_ids = _reference_ranking(index, query, bm25_ids) if mode == "hybrid" else []
    if mode == "hybrid":
        ranked, rrf = _fuse([vector_ids, bm25_ids, reference_ids])
    else:
        ranked, rrf = (vector_ids or bm25_ids), {}

    pool = _take(ranked, index.chunks, named, config.RERANK_CANDIDATES if rerank else k)
    search_pos = {chunk_id: rank for rank, chunk_id in enumerate(pool, start=1)}
    rerank_by_id = {}
    if rerank and pool:
        rerank_by_id = dict(zip(pool, rerank_scores(query, [index.chunks[i] for i in pool])))
        pool = _take(sorted(pool, key=rerank_by_id.get, reverse=True), index.chunks, named, k)

    vector_pos = {chunk_id: rank for rank, chunk_id in enumerate(vector_ids, start=1)}
    bm25_pos = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    results = []
    for chunk_id, cosine in zip(pool, index.cosine(query_vec, pool) if pool else [], strict=True):
        result = dict(index.chunks[chunk_id])
        result.update(score=float(cosine), rerank_score=rerank_by_id.get(chunk_id),
                      vector_rank=vector_pos.get(chunk_id), bm25_rank=bm25_pos.get(chunk_id),
                      search_rank=search_pos.get(chunk_id), bm25=float(bm25_scores[chunk_id]),
                      reference=chunk_id in reference_ids, rrf=rrf.get(chunk_id))
        results.append(result)
    return results


def merge_results(result_lists: list[list[dict]], k: int) -> list[dict]:
    """Merge the results of several queries (query expansion) with Reciprocal Rank Fusion.
    A chunk keeps its entry from the first list, so the original question's scores are kept."""
    fused, first_entry = {}, {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            fused[result["id"]] = fused.get(result["id"], 0.0) + 1.0 / (config.RRF_K + rank)
            first_entry.setdefault(result["id"], result)
    return [first_entry[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


def is_reranked(results: list[dict]) -> bool:
    return any(r.get("rerank_score") is not None for r in results)


def score_name(results: list[dict]) -> str:
    return "reranker score" if is_reranked(results) else "similarity"


def threshold(results: list[dict]) -> float:
    return config.RERANK_MIN_SCORE if is_reranked(results) else config.MIN_SCORE


def best_score(results: list[dict]) -> float | None:
    """The score the refusal threshold is checked against: the best cross-encoder score when
    the results were reranked, otherwise the best cosine similarity."""
    key = "rerank_score" if is_reranked(results) else "score"
    return max((r[key] for r in results if r.get(key) is not None), default=None)


def exact_terms(query: str) -> set[str]:
    """Codes, acronyms and references in the question ('12.3', 'ECTS', 'article_9').
    Plain numbers like a year are left out: they appear everywhere."""
    words = re.findall(r"\w+(?:[.\-/]\w+)*", query)
    codes = {w.lower() for w in words
             if len(w) >= 2 and ((re.search(r"\d", w) and not w.isdigit()) or (w.isupper() and w.isalpha()))}
    return codes | reference_tokens(query)


def is_confident(results: list[dict], query: str = "") -> bool:
    """False when even the best passage is unrelated (the search always returns k results,
    even for "What's the weather?"). A passage containing an exact code from the question
    also counts, since "What does Article 12.3 say?" can score low against a long chunk."""
    best = best_score(results)
    if best is not None and best >= threshold(results):
        return True
    terms = exact_terms(query)
    return bool(terms) and any(terms & set(tokenize(r["text"])) for r in results)


_QUESTION_WORDS = set("""a an the is are was were be been being am to of in on for and or but if by with as at it its
this that these those does do did done what which who whom whose how when where why can could should would will shall
may might must not no nor from into onto than then there their they them he she his her we our you your i me my about
above under over between both each every any all some such only also too very just so up out off again once here same
other another nothing middle tell give say says said use using used according uploaded please explain describe list
name named mean means document documents pdf page pages passage passages recital recitals version""".split())


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset:
    return frozenset().union(*get_index().token_sets)


def _in_documents(word: str, vocabulary: frozenset) -> bool:
    """The word, a simple inflection of it, or (for long words) its first 6 letters is in the corpus."""
    if word in vocabulary:
        return True
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"),
                                ("ed", ""), ("ed", "e"), ("ly", "")):
        stem = word[:len(word) - len(suffix)]
        if word.endswith(suffix) and len(stem) >= 3 and stem + replacement in vocabulary:
            return True
    return len(word) >= 7 and any(v.startswith(word[:6]) for v in vocabulary)


def unknown_words(query: str) -> list[str]:
    """Words of the question found nowhere in the documents ('shirt', 'fifa'). A hint for the
    model: "the fine for wearing a blue shirt" is close to real passages about fines."""
    vocabulary = _vocabulary()
    words = dict.fromkeys(re.findall(r"[a-z]+", query.lower()))
    return [w for w in words if len(w) >= 3 and w not in _QUESTION_WORDS and not _in_documents(w, vocabulary)]


def unknown_words_note(query: str) -> str:
    missing = unknown_words(query)
    if not missing:
        return ""
    return (f"Words of the question that appear nowhere in the documents: {', '.join(missing)}. "
            "If the question is about one of them, the documents don't cover it.")
