

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QUESTIONS_FILE = DATA_DIR / "questions.json"
RESULTS_FILE = DATA_DIR / "eval_results.json"
AGENT_RESULTS_FILE = DATA_DIR / "eval_results_agent.json"

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided "
    "documents to answer this question."
)

# Questions with no expected pages: the right behaviour is to refuse.
REFUSAL_CATEGORIES = {"out-of-scope"}

# Use the same code as the app.
sys.path.insert(0, str(PROJECT_ROOT / "app"))
import retrieval as retrieval_module  # noqa: E402
from retrieval import TOP_K, retrieve  # noqa: E402


# ---------- Checks ----------

def is_refusal(answer: str) -> bool:
    """True when the model gave the required fallback message."""
    return FALLBACK_MESSAGE.lower() in answer.strip().strip('"').lower()


def retrieval_check(entry: dict, results: list[dict]) -> dict:
    """Compare expected pages with what retrieval returned."""
    expected = {(c["source"], c["page"]) for c in entry.get("expected_chunks") or []}
    retrieved = [(c["source"], c["page"]) for c in results]

    if not expected:
        return {"expected": [], "found": [], "hit": None, "all_found": None}

    found = [key for key in expected if key in retrieved]
    first_rank = min(
        (retrieved.index(key) + 1 for key in found), default=None
    )

    return {
        "expected": [list(key) for key in sorted(expected)],
        "found": [list(key) for key in sorted(found)],
        "hit": bool(found),                   # at least one expected page
        "all_found": len(found) == len(expected),
        "first_hit_rank": first_rank,
    }


def judge_answer(question: str, expected: str, answer: str) -> dict:
    """LLM-as-judge: compare the answer with the expected answer."""
    from generate_answer import client
    from cost_tracker import track_cost

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Compare the system's answer "
                    "with the expected answer. Verdict 'correct' if it contains "
                    "the key facts of the expected answer and nothing that "
                    "contradicts it, 'partial' if it has some key facts but "
                    "misses others, 'incorrect' if it is wrong, refuses, or "
                    "misses the main point. Never mark an answer partial or "
                    "incorrect only because it contains extra information. "
                    "Ignore citation markers like [1] and (file.pdf, p. N). "
                    f"Today's date is {date.today():%d %B %Y}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Expected answer: {expected}\n\n"
                    f"System answer: {answer}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["correct", "partial", "incorrect"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["verdict", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    )
    track_cost(response)
    return json.loads(response.choices[0].message.content)


def decide(entry: dict, retrieval: dict, answer: str | None,
           judgement: dict | None) -> tuple[str, str]:
    """Return (status, reason). Status is pass, fail or review."""
    category = entry["category"]

    # Out-of-scope: the system must refuse.
    if category in REFUSAL_CATEGORIES:
        if answer is None:
            return "review", "retrieval only: refusal not checked"
        if is_refusal(answer):
            return "pass", "correctly refused"
        return "fail", "answered an out-of-scope question"

    # Ambiguous: no single right answer, a person should look at it.
    if category == "ambiguous":
        if answer is None:
            return "review", "retrieval only"
        return "review", "ambiguous question: check the answer by hand"

    # In-scope questions: retrieval first.
    if retrieval["hit"] is False:
        return "fail", "retrieval miss: expected pages not in top-k"

    if answer is None:
        return "pass", "retrieval hit (answer not generated)"

    if is_refusal(answer):
        return "fail", "false refusal: pages were retrieved but model refused"

    if judgement is None:
        return "review", "retrieval hit: check the answer by hand"

    if judgement["verdict"] == "correct":
        return "pass", "judge: correct"
    if judgement["verdict"] == "partial":
        return "fail", f"generation error (partial): {judgement['reason']}"
    return "fail", f"generation error: {judgement['reason']}"


# ---------- Agent evaluation ----------

def pages_seen_by_agent(run: dict) -> list[dict]:
    """Every (source, page) the agent's tools returned, plus its citations.

    Used as the agent's "retrieved chunks": if an expected page is here,
    the agent found the right evidence.
    """
    seen = []

    def add(source, page):
        item = {"source": source, "page": page, "score": None}
        if source and page and item not in seen:
            seen.append(item)

    for step in run.get("steps", []):
        result = step.get("result") or {}
        if not result.get("ok"):
            continue
        for item in result.get("results", []):          # search_corpus
            add(item.get("source"), item.get("page"))
        if step["tool"] == "quote_exact":
            add(result.get("source"), result.get("page"))
        if step["tool"] == "get_article":
            for page in result.get("pages", []):
                add(result.get("source"), page)

    for citation in run.get("citations", []):
        add(citation["source"], citation["page"])

    return seen


def tools_check(entry: dict, run: dict) -> dict | None:
    """Did the agent use the tools a multi-step question needs?"""
    expected = entry.get("expected_tools")
    if not expected:
        return None
    used = {step["tool"] for step in run.get("steps", [])}
    missing = [tool for tool in expected if tool not in used]
    return {"expected": expected, "used": sorted(used), "missing": missing}


# ---------- Scorecard ----------

def rate(values: list) -> str:
    """Format a hit rate like '12/15 (80%)', ignoring None."""
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    hits = sum(values)
    return f"{hits}/{len(values)} ({hits / len(values):.0%})"


def build_summary(rows: list[dict]) -> dict:
    """Overall, per-category and per-difficulty numbers."""
    def block(group: list[dict]) -> dict:
        return {
            "questions": len(group),
            "retrieval_hit": rate([r["retrieval"]["hit"] for r in group]),
            "all_expected_found": rate([r["retrieval"]["all_found"] for r in group]),
            "pass": sum(r["status"] == "pass" for r in group),
            "fail": sum(r["status"] == "fail" for r in group),
            "review": sum(r["status"] == "review" for r in group),
        }

    by_category = defaultdict(list)
    by_difficulty = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_difficulty[row["difficulty"]].append(row)

    return {
        "overall": block(rows),
        "by_category": {k: block(v) for k, v in by_category.items()},
        "by_difficulty": {k: block(v) for k, v in by_difficulty.items()},
    }


def print_summary(summary: dict, rows: list[dict]) -> None:
    """Print the scorecard to the terminal."""
    def line(name: str, b: dict) -> None:
        print(f"  {name:<16} n={b['questions']:<3} "
              f"retrieval hit {b['retrieval_hit']:<13} "
              f"pass {b['pass']:<3} fail {b['fail']:<3} review {b['review']}")

    print("\n================ SCORECARD ================")
    line("OVERALL", summary["overall"])
    print("\nBy category:")
    for name, b in summary["by_category"].items():
        line(name, b)
    print("\nBy difficulty:")
    for name in ("easy", "medium", "hard"):
        if name in summary["by_difficulty"]:
            line(name, summary["by_difficulty"][name])

    failed = [r for r in rows if r["status"] == "fail"]
    print(f"\nFailed questions ({len(failed)}):")
    for r in failed:
        print(f"  #{r['id']:<3} [{r['category']}] {r['question'][:70]}")
        print(f"        → {r['reason']}")

    review = [r for r in rows if r["status"] == "review"]
    if review:
        print(f"\nTo check by hand ({len(review)}): "
              + ", ".join(f"#{r['id']}" for r in review))


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the evaluation questions and print a scorecard."
    )
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip answer generation (cheapest)")
    parser.add_argument("--judge", action="store_true",
                        help="score answers with gpt-4o-mini as judge")
    parser.add_argument("--priority", action="store_true",
                        help="only run the priority questions")
    parser.add_argument("-k", type=int, default=TOP_K, help="chunks to retrieve")
    parser.add_argument("--search", choices=["hybrid", "vector"],
                        default=retrieval_module.SEARCH_MODE,
                        help="retrieval mode to evaluate (default: hybrid)")
    parser.add_argument("--agent", action="store_true",
                        help="run each question through the agent (agent.py)")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE),
                        help="questions file (default: data/questions.json)")
    args = parser.parse_args()

    retrieval_module.SEARCH_MODE = args.search

    if args.agent:
        from agent import run_agent
        args.retrieval_only = False
    elif not args.retrieval_only:
        from generate_answer import generate_answer

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    selected = [
        (i, q) for i, q in enumerate(questions)
        if not args.priority or q.get("priority")
    ]

    rows = []
    for number, (i, entry) in enumerate(selected, start=1):
        print(f"[{number}/{len(selected)}] #{i} {entry['question'][:70]}")
        start = time.time()

        run = None
        answer = None
        judgement = None

        if args.agent:
            # The agent chooses its own tools; its "retrieved chunks" are
            # the pages its tools returned and the pages it cited.
            run = run_agent(entry["question"])
            results = pages_seen_by_agent(run)
            answer = run["answer"] or ""
        else:
            # Fixed RAG pipeline: retrieve, then (optionally) generate.
            results = retrieve(entry["question"], k=args.k)
            if not args.retrieval_only:
                answer = generate_answer(entry["question"], results)

        retrieval = retrieval_check(entry, results)

        if (answer is not None and args.judge
                and entry.get("expected_answer") and not is_refusal(answer)):
            judgement = judge_answer(
                entry["question"], entry["expected_answer"], answer
            )

        status, reason = decide(entry, retrieval, answer, judgement)

        tools = tools_check(entry, run) if run else None
        if tools and tools["missing"] and status == "pass":
            status, reason = "fail", f"agent did not use: {', '.join(tools['missing'])}"

        rows.append({
            "id": i,
            "question": entry["question"],
            "category": entry["category"],
            "sub_type": entry.get("sub_type"),
            "difficulty": entry["difficulty"],
            "priority": entry.get("priority", False),
            "retrieval": retrieval,
            "top_score": max((c["score"] for c in results
                              if c.get("score") is not None), default=None),
            "retrieved": [
                {"source": c["source"], "page": c["page"],
                 "score": (round(c["score"], 4)
                           if c.get("score") is not None else None)}
                for c in results
            ],
            "answer": answer,
            "expected_answer": entry.get("expected_answer"),
            "judge": judgement,
            "status": status,
            "reason": reason,
            "seconds": round(time.time() - start, 2),
            "agent": None if run is None else {
                "tools": [step["tool"] for step in run["steps"]],
                "tools_check": tools,
                "model_calls": run["model_calls"],
                "stop_reason": run["stop_reason"],
            },
        })

    summary = build_summary(rows)
    print_summary(summary, rows)

    results_file = AGENT_RESULTS_FILE if args.agent else RESULTS_FILE
    # One results file per questions file, so runs don't overwrite each other.
    stem = Path(args.questions).stem
    if stem != "questions":
        results_file = results_file.with_name(f"{results_file.stem}_{stem}.json")
    results_file.write_text(
        json.dumps(
            {
                "run_at": datetime.now().isoformat(timespec="seconds"),
                "settings": {
                    "top_k": args.k,
                    "search": args.search,
                    "agent": args.agent,
                    "questions": Path(args.questions).name,
                    "retrieval_only": args.retrieval_only,
                    "judge": args.judge,
                    "priority_only": args.priority,
                },
                "summary": summary,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {results_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
