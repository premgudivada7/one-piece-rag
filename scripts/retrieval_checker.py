"""Shows what retrieval finds for one question, and whether the expected pages came back.

    python scripts/retrieval_checker.py --id q07
    python scripts/retrieval_checker.py "What does GOVERN 1.1 say?" --compare
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.chunking import page_label
from app.evaluation import load_questions, retrieval_check
from app.retrieval import best_score, exact_terms, is_confident, retrieve, score_name, threshold


def find_question(text: str | None, qid: str | None) -> dict | None:
    try:
        questions = load_questions()
    except FileNotFoundError:
        return None
    return next((q for q in questions if (qid and q["id"] == qid)
                 or (text and q["question"].strip().lower() == text.strip().lower())), None)


def show(question: str, q: dict | None, k: int, mode: str, rerank: bool, width: int):
    results = retrieve(question, k=k, mode=mode, rerank=rerank)
    check = retrieval_check(q, results) if q else None
    expected = set(check["expected"]) if check else set()
    best = best_score(results)
    verdict = "confident" if is_confident(results, question) else "WOULD REFUSE"
    print(f"\n=== {mode}{' + reranking' if rerank else ''}  k={k}  best {score_name(results)} "
          f"{best if best is None else round(best, 3)} (threshold {threshold(results)}: {verdict})")
    if exact_terms(question):
        print(f"    exact terms in the question: {', '.join(sorted(exact_terms(question)))}")
    for rank, r in enumerate(results, start=1):
        star = "⭐" if any(r["source"].lower().endswith(s) and p in r["pages"] for s, p in expected) else "  "
        rerank_text = f"rerank={r['rerank_score']:.2f} " if r.get("rerank_score") is not None else ""
        print(f"{star} {rank}. [{r['source']} {page_label(r)}] {rerank_text}faiss_cos={r['score']:.3f} "
              f"search#{r['search_rank']} vec#{r['vector_rank'] or '-'} bm25#{r['bm25_rank'] or '-'} (chunk {r['id']})")
        print("      " + r["text"][:width].replace("\n", " ") + ("…" if len(r["text"]) > width else ""))
    if check and check["expected"]:
        for pair in check["expected"]:
            print(f"   {'✅' if pair in check['found'] else '❌'} expected {pair[0]} p.{pair[1]}")
        print(f"   {'HIT' if check['hit'] else 'MISS'} · first expected chunk at rank {check['first_rank'] or '-'}")
    elif q:
        print(f"   (no expected_chunks: category {q.get('category')})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--id", help="question id from questions.json, e.g. q07")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--compare", action="store_true", help="vector only, vector + rerank, hybrid + rerank")
    parser.add_argument("--width", type=int, default=220, help="characters of each chunk to print")
    args = parser.parse_args()

    q = find_question(args.question, args.id)
    if args.id and not q:
        sys.exit(f"No question with id {args.id} in data/questions.json")
    question = args.question or (q["question"] if q else None)
    if not question:
        parser.print_help()
        return
    print(f"Q: {question}")
    if q:
        print(f"   {q['id']} · {q.get('category')} {q.get('sub_type') or ''} · expected: {q.get('expected_answer')}")
    runs = ([("vector", False), ("vector", True), ("hybrid", True)] if args.compare
            else [(args.mode, config.USE_RERANK and not args.no_rerank)])
    for mode, rerank in runs:
        show(question, q, args.k, mode, rerank, args.width)


if __name__ == "__main__":
    main()
