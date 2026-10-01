"""Checks data/questions.json against the Assignment 1 checklist, the follow-up questions
(RAG day 2) and the agent questions (Agent day 2).

    python scripts/validate_questions.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.evaluation import AGENT_SCENARIOS, CATEGORIES, CROSS_SUBTYPES, STRESS_SUBTYPES, load_questions
from app.tools import TOOL_NAMES

MINIMUMS = {"factual": 10, "cross-reference": 8, "stress-test": 10, "out-of-scope": 3, "ambiguous": 2}


def main():
    questions = load_questions()
    problems, warnings = [], []

    def need(ok, message, warn=False):
        if not ok:
            (warnings if warn else problems).append(message)

    follow_ups = [q for q in questions if q.get("history") and q.get("category") != "agent"]
    agent = [q for q in questions if q.get("category") == "agent"]
    core = [q for q in questions if q not in follow_ups and q not in agent]
    need(40 <= len(core) <= 50, f"{len(core)} questions without follow-ups and agent questions (expected 40 to 50)")
    categories = Counter(q.get("category") for q in core)
    for category, minimum in MINIMUMS.items():
        need(categories[category] >= minimum, f"{categories[category]} {category} questions (minimum {minimum})")
    for category in Counter(q.get("category") for q in questions):
        need(category in CATEGORIES, f"unknown category {category!r} (allowed: {', '.join(CATEGORIES)})")

    cross = {q.get("sub_type") for q in core if q.get("category") == "cross-reference"}
    stress = {q.get("sub_type") for q in core if q.get("category") == "stress-test"}
    need(len(cross & set(CROSS_SUBTYPES)) >= 5, f"cross-reference covers {len(cross & set(CROSS_SUBTYPES))} "
         f"sub-types (need 5 of 6); missing: {', '.join(sorted(set(CROSS_SUBTYPES) - cross))}")
    need(len(stress & set(STRESS_SUBTYPES)) >= 8, f"stress-test covers {len(stress & set(STRESS_SUBTYPES))} "
         f"sub-types (need 8 of 11); missing: {', '.join(sorted(set(STRESS_SUBTYPES) - stress))}")
    need(sum(bool(q.get("priority")) for q in questions) == 5,
         f"{sum(bool(q.get('priority')) for q in questions)} priority questions (need exactly 5)")
    need(3 <= len(follow_ups) <= 4, f"{len(follow_ups)} follow-up questions with a history field (need 3 to 4)", warn=True)
    need(any(q.get("sub_type") == "on_topic_unanswerable" for q in questions),
         "no on_topic_unanswerable question: the refusal threshold is tuned on these and out-of-scope ones", warn=True)
    missing_scenarios = [s for s in AGENT_SCENARIOS if s not in {q.get("sub_type") for q in agent}]
    need(not missing_scenarios, f"agent scenarios without a question: {', '.join(missing_scenarios)}", warn=True)

    pages = set()
    if config.CORPUS_PATH.exists():
        pages = {(e["source"], int(e["page"])) for e in json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))}
    else:
        warnings.append("data/corpus.json not found: source/page values were not checked")

    for q in questions:
        tag = f"{q['id']} ({q.get('category')})"
        need(q.get("difficulty") in ("easy", "medium", "hard"), f"{tag}: difficulty must be easy/medium/hard")
        if q.get("category") == "cross-reference":
            need(q.get("sub_type") in CROSS_SUBTYPES, f"{tag}: invalid or missing sub_type {q.get('sub_type')!r}")
        if q.get("category") == "stress-test":
            need(q.get("sub_type") in STRESS_SUBTYPES, f"{tag}: invalid or missing sub_type {q.get('sub_type')!r}")
        if q.get("category") == "agent":
            need(q.get("sub_type") in AGENT_SCENARIOS, f"{tag}: sub_type must be one of {', '.join(AGENT_SCENARIOS)}")
            need(isinstance(q.get("expected_tools"), list), f"{tag}: expected_tools must be a list ([] for no_tool)")
            for tool in q.get("expected_tools") or []:
                need(all(t in TOOL_NAMES for t in tool.split("|")), f"{tag}: unknown tool {tool!r}")
            need(q.get("expected_answer"), f"{tag}: missing expected_answer")
        if "history" in q:
            need(isinstance(q["history"], list) and all(isinstance(h, str) for h in q["history"]),
                 f"{tag}: history must be a list of earlier questions")
        if q.get("category") in ("factual", "cross-reference", "stress-test") and q.get("sub_type") != "on_topic_unanswerable":
            for field in ("expected_answer", "source", "page", "expected_chunks"):
                need(q.get(field) is not None, f"{tag}: missing {field}")
        for chunk in q.get("expected_chunks") or []:
            if pages:
                need((chunk.get("source"), int(chunk.get("page", -1))) in pages,
                     f"{tag}: expected chunk {chunk.get('source')} p.{chunk.get('page')} is not in corpus.json")

    print(f"{len(questions)} questions: {len(core)} core, {len(follow_ups)} follow-ups, {len(agent)} agent")
    for category in CATEGORIES:
        print(f"  {category:<16} {Counter(q.get('category') for q in questions)[category]:>3}")
    print(f"  cross-reference sub-types: {', '.join(sorted(s for s in cross if s))}")
    print(f"  stress-test sub-types    : {', '.join(sorted(s for s in stress if s))}")
    for w in warnings:
        print(f"⚠️  {w}")
    for p in problems:
        print(f"❌ {p}")
    print("✅ questions.json passes the checklist" if not problems else f"\n{len(problems)} problem(s) to fix")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
