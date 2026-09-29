"""Inspect retrieval for one question.

Shows the top-k chunks (source, page, score, text) and, when the
question is in data/questions.json, whether the expected pages were
retrieved (✅) or missed (❌).

Usage:
    python scripts/retrieval_checker.py "What is a high-risk AI system?"
    python scripts/retrieval_checker.py --id 3        # question 3 of questions.json
    python scripts/retrieval_checker.py --id 3 -k 5   # only look at the top 5

Cost: one embedding call per run (a tiny fraction of a cent).
"""

import argparse
import json
import sys
from pathlib import Path


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "data" / "questions.json"
PREVIEW_CHARS = 300

# Let this script use the same retrieve() as the app.
sys.path.insert(0, str(PROJECT_ROOT / "app"))
from retrieval import TOP_K, retrieve  # noqa: E402


# ---------- Helpers ----------

def load_questions() -> list[dict]:
    """Read the evaluation questions."""
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def find_question(text: str, questions: list[dict]) -> dict | None:
    """Return the questions.json entry with exactly this text, if any."""
    for entry in questions:
        if entry["question"].strip().lower() == text.strip().lower():
            return entry
    return None


def expected_pages(entry: dict | None) -> set[tuple[str, int]]:
    """Expected (source, page) pairs, or an empty set if none."""
    if not entry or not entry.get("expected_chunks"):
        return set()
    return {(c["source"], c["page"]) for c in entry["expected_chunks"]}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", help="question text")
    parser.add_argument("--id", type=int, help="index in questions.json (0-based)")
    parser.add_argument("-k", type=int, default=TOP_K, help="chunks to retrieve")
    args = parser.parse_args()

    questions = load_questions()

    if args.id is not None:
        entry = questions[args.id]
        question = entry["question"]
    elif args.question:
        question = args.question
        entry = find_question(question, questions)
    else:
        parser.error("give a question or --id")

    expected = expected_pages(entry)
    results = retrieve(question, k=args.k)

    print(f"\nQuestion: {question}")
    if entry:
        print(f"Category: {entry['category']} | Difficulty: {entry['difficulty']}")
    print(f"Top-k: {args.k}\n")

    # ---- Retrieved chunks ----
    for rank, chunk in enumerate(results, start=1):
        key = (chunk["source"], chunk["page"])
        mark = "✅" if key in expected else "  "
        preview = chunk["text"][:PREVIEW_CHARS].replace("\n", " ")

        print(f"{mark} [{rank}] {chunk['source']} — page {chunk['page']}"
              f" — score {chunk['score']:.4f}")
        print(f"      {preview}…\n")

    # ---- Expected chunks: found or missed ----
    if not entry:
        print("Question not found in questions.json, so there is nothing to check.")
        return

    if not expected:
        print("No expected chunks for this question "
              f"({entry['category']}): the answer should be a refusal.")
        print(f"Top score: {results[0]['score']:.4f} "
              "(a low score suggests the corpus does not cover it)")
        return

    retrieved = {(c["source"], c["page"]): rank
                 for rank, c in reversed(list(enumerate(results, start=1)))}

    print("Expected chunks:")
    for source, page in sorted(expected):
        rank = retrieved.get((source, page))
        if rank:
            print(f"  ✅ {source} page {page} (first found at rank {rank})")
        else:
            print(f"  ❌ {source} page {page} (missed)")

    found = sum(1 for key in expected if key in retrieved)
    print(f"\nFound {found}/{len(expected)} expected pages.")


if __name__ == "__main__":
    main()
