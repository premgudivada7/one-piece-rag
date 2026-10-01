"""Recovery check: the first search of each run is forced to fail with the real
"No matching passages found" message. The agent must notice, rephrase (or use another tool)
and still answer with a citation, without repeating the same call.
(The prompt injection test is in scripts/injection_test.py.)

    python scripts/agent_checks.py
    python scripts/agent_checks.py --ids q10,q11,q12 --runs 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agent, config
from app.evaluation import load_questions
from app.llm import load_costs
from app.messaging import fake_send
from app.rag import cited_sources, is_refusal

WEAK_SEARCH = ("No matching passages found (best reranker score -9.80, threshold -5.0). "
               "Try different keywords, or tell the user the documents don't cover this.")
real_search = agent.search_corpus


def recovery_check(question: str) -> dict:
    calls = {"n": 0}

    def failing_first(query, seen_ids=None, k=None):
        calls["n"] += 1
        return (WEAK_SEARCH, []) if calls["n"] == 1 else real_search(query, seen_ids, k)

    agent.search_corpus = failing_first
    try:
        run = agent.AgentRun(question)
        answer = run.run(approve=fake_send)
    finally:
        agent.search_corpus = real_search
    tools = [e["tool"] for e in run.tool_events()]
    searches = run.searches()
    rephrased = len(searches) >= 2 and searches[1].strip().lower() != searches[0].strip().lower()
    other_tool = any(t in ("get_article", "quote_exact") for t in tools[1:])
    cited = bool(cited_sources(answer or ""))
    return {"question": question, "searches": searches, "tools": tools, "rephrased": rephrased,
            "used_other_tool": other_tool, "repeated_calls": run.summary()["repeated_calls"],
            "steps": run.steps_used, "hit_step_cap": run.hit_step_cap, "cited": cited,
            "recovered": (rephrased or other_tool) and cited and not is_refusal(answer or ""),
            "answer": answer, "trace_id": run.id}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default="q10,q11,q12")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    wanted = [i.strip() for i in args.ids.split(",")]
    questions = [q["question"] for q in load_questions() if q["id"] in wanted]
    config.SEND_BACKEND = "dry_run"
    cost_before = load_costs()["total"]

    rows = [recovery_check(q) for _ in range(args.runs) for q in questions]
    print("\n| Question | Next action | Repeated calls | Steps | Cited | Recovered |")
    print("|---|---|:---:|:---:|:---:|:---:|")
    for r in rows:
        if r["rephrased"]:
            action = "rephrased: " + r["searches"][1][:40]
        elif r["used_other_tool"]:
            action = "other tool: " + ", ".join(t for t in r["tools"][1:] if t != "search_corpus")[:40]
        else:
            action = "gave up"
        print(f"| {r['question'][:50]} | {action} | {r['repeated_calls']} | {r['steps']} | "
              f"{'✅' if r['cited'] else '❌'} | {'✅' if r['recovered'] else '❌'} |")
    print(f"Recovered {sum(r['recovered'] for r in rows)}/{len(rows)}")

    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": config.CHAT_MODEL, "recovery": rows,
           "cost": round(load_costs()["total"] - cost_before, 6)}
    path = config.DATA_DIR / "agent_checks.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncost ${out['cost']:.4f} · saved {path.relative_to(config.ROOT)} · traces in data/traces/")


if __name__ == "__main__":
    main()
