"""Runs questions.json through the system and prints a scorecard.

    python scripts/eval_runner.py --mode retrieval            recall@1/3/5, MRR, latency, threshold check (free)
    python scripts/eval_runner.py --ablation                  ablation table, retrieval only
    python scripts/eval_runner.py --judge                     full RAG answers, judged
    python scripts/eval_runner.py --priority --judge          the 5 priority questions first
    python scripts/eval_runner.py --mode agent --judge        agent questions + priority questions
    python scripts/eval_runner.py --judge --compare data/eval_runs/<previous>_rag.json    regression check
"""
import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.agent import AgentRun
from app.chunking import page_label
from app.evaluation import (RECALL_KS, answer_overlap, expected_pairs, judge, load_questions, mean_text, pct,
                            question_set, rag_metrics, ranking_metrics, retrieval_check, should_refuse)
from app.llm import load_costs
from app.messaging import fake_send
from app.rag import answer_question, cited_sources, search
from app.retrieval import retrieve

ICON = {"pass": "✅", "fail": "❌", "unjudged": "❔", "error": "💥"}
ABLATION = [
    ("Vector only", {"mode": "vector", "rerank": False, "rewrite": False, "expand": False}),
    ("+ reranking (cross-encoder)", {"mode": "vector", "rerank": True, "rewrite": False, "expand": False}),
    ("+ BM25 and exact references (hybrid)", {"mode": "hybrid", "rerank": True, "rewrite": False, "expand": False}),
    ("+ query rewriting (follow-ups)", {"mode": "hybrid", "rerank": True, "rewrite": True, "expand": False}),
    ("+ query expansion (optional)", {"mode": "hybrid", "rerank": True, "rewrite": True, "expand": True}),
]


def mark(value) -> str:
    return "✅" if value else "❌" if value is False else "-"


def answerable(q: dict) -> bool:
    """Has expected pages to find, and is not a question the system should refuse."""
    return bool(expected_pairs(q)) and not should_refuse(q)


_history_cache: dict = {}


def replay_history(q: dict, mode: str) -> list[dict] | None:
    """For a follow-up question, replay its earlier user questions so the system has the
    conversation. Retrieval-only runs replay the questions alone (no answer is generated)."""
    history = []
    for earlier in q.get("history") or []:
        history.append({"role": "user", "content": earlier})
        if mode == "retrieval":
            continue
        key = (mode, tuple(m["content"] for m in history))
        if key not in _history_cache:
            _history_cache[key] = (AgentRun(earlier, history=history[:-1]).run(approve=fake_send) if mode == "agent"
                                   else answer_question(earlier, history=history[:-1])["answer"])
        history.append({"role": "assistant", "content": _history_cache[key]})
    return history or None


def run_one(q: dict, mode: str, use_judge: bool, k: int | None, use_metrics: bool = False,
            search_options: dict | None = None) -> dict:
    record = {key: q.get(key) for key in ("id", "question", "category", "sub_type", "difficulty", "priority",
                                          "expected_answer", "expected_tools", "history")}
    record.update(mode=mode, answerable=answerable(q))
    answer, agent_info, cost, calls, search_query, evidence = None, None, 0.0, 0, q["question"], None
    try:
        history = replay_history(q, mode)
        start = time.time()
        if mode == "retrieval":
            found = search(q["question"], history, k=k, **(search_options or {}))
            results, refused, top = found["results"], not found["confident"], found["top_score"]
            cost, calls, search_query = found["cost"], found["model_calls"], found["search_query"]
        elif mode == "rag":
            out = answer_question(q["question"], history=history, k=k)
            results, answer, refused, top = out["results"], out["answer"], out["refused"], out["top_score"]
            cost, calls, search_query = out["cost"], out["model_calls"], out["search_query"]
        else:
            run = AgentRun(q["question"], history=history)
            answer = run.run(approve=fake_send)
            summary = run.summary()
            results, refused, top, cost, calls = run.sources, summary["refused"], None, summary["cost"], summary["steps"]
            # calculator and date_tool results are evidence too, for the faithfulness judge
            evidence = results + [{"source": e["tool"], "page": "-", "text": e["result"]}
                                  for e in run.tool_events() if e["tool"] in ("calculator", "date_tool")]
            agent_info = {key: summary[key] for key in ("id", "steps", "hit_step_cap", "tools_used", "searches",
                                                        "repeated_calls", "send_attempts", "send_guard_blocks")}
            # for "act, with approval" the outcome is the message it proposed, so the judge sees it too
            agent_info["proposed_messages"] = [e["args"].get("text", "") for e in run.tool_events("send_message")]
        latency = round(time.time() - start, 2)
    except Exception as e:
        record.update(error=f"{type(e).__name__}: {e}", passed=False, outcome="error",
                      failure_reason=f"crash: {type(e).__name__}", latency_s=None)
        return record

    check = retrieval_check(q, results)
    record.update(
        search_query=search_query if search_query != q["question"] else None,
        retrieved=[{"source": r["source"], "label": page_label(r), "score": r.get("score"),
                    "rerank_score": r.get("rerank_score")} for r in results],
        retrieval=check, top_score=top, refused=refused, answer=answer,
        has_citation=bool(cited_sources(answer)) if answer and results else None,  # only answers that used the documents
        overlap=answer_overlap(q.get("expected_answer"), answer) if answer else None,
        agent=agent_info, cost=cost, model_calls=calls, judge=None, metrics=None, latency_s=latency)
    if use_metrics and mode != "retrieval" and answer and not refused:
        record["metrics"] = rag_metrics(q["question"], answer, evidence or results)

    passed, reason = None, None
    if should_refuse(q):
        passed = refused
        if not passed:
            reason = "search confident on a question it should refuse" if mode == "retrieval" else "answered instead of refusing"
    elif mode == "retrieval":
        passed = check["hit"]
        reason = None if passed is not False else "retrieval miss"
    else:
        if use_judge:
            proposals = (agent_info or {}).get("proposed_messages") or []
            judged = (answer or "") + "".join(f"\n\nMessage proposed for approval: {p}" for p in proposals)
            record["judge"] = judge(q, judged)
            verdict = record["judge"]["verdict"]
            passed = None if verdict == "unclear" else verdict == "correct"
        if q.get("category") == "ambiguous":
            reason = "ambiguity not handled" if passed is False else None
        elif refused and passed is not True and q.get("sub_type") != "no_tool":  # the judge may accept a justified refusal
            passed, reason = False, "wrong refusal" + (" (retrieval miss)" if check["hit"] is False else "")
        elif passed is False:
            reason = ("retrieval miss" if check["hit"] is False else "generation error") + (
                " (partial answer)" if record["judge"]["verdict"] == "partial" else "")
    if agent_info and q.get("expected_tools") is not None:
        # "search_corpus|get_article" means either tool is fine; [] means no tool at all
        missing = [t for t in q["expected_tools"] if not any(alt in agent_info["tools_used"] for alt in t.split("|"))]
        unwanted = not q["expected_tools"] and bool(agent_info["tools_used"])
        agent_info["expected_tools_used"] = not missing and not unwanted
        if missing or unwanted:
            reason = (reason + "; " if reason else "") + (f"did not use {', '.join(missing)}" if missing else "used a tool")
    if agent_info and should_refuse(q):
        agent_info["stopped_cleanly"] = bool(refused) and agent_info["searches"] <= 2 and not agent_info["hit_step_cap"]
    record.update(passed=passed, failure_reason=reason,
                  outcome="unjudged" if passed is None else ("pass" if passed else "fail"))
    return record


def ranking_line(records: list[dict]) -> str:
    m = ranking_metrics([r.get("retrieval") for r in records])
    if not m["n"]:
        return "-"
    recalls = " · ".join(f"recall@{k} {m[f'recall@{k}']:.2f}" for k in RECALL_KS)
    return f"{recalls} · MRR {m['mrr']:.2f} · full recall {m['full_recall']:.2f} (n={m['n']})"


def group_table(records: list[dict], key: str, title: str):
    groups = defaultdict(list)
    for r in records:
        groups[r.get(key) or "-"].append(r)
    print(f"\n{title}")
    print(f"  {'':<26} {'n':>3}  {'pass':<12} {'recall@5':>8} {'MRR':>6}")
    for name in sorted(groups, key=str):
        rows = groups[name]
        m = ranking_metrics([r.get("retrieval") for r in rows])
        r5 = f"{m['recall@5']:.2f}" if m["n"] else "-"
        mrr = f"{m['mrr']:.2f}" if m["n"] else "-"
        print(f"  {str(name):<26} {len(rows):>3}  {pct([r['passed'] for r in rows]):<12} {r5:>8} {mrr:>6}")


def calibration(records: list[dict]):
    """Help choose the refusal threshold: best score of answerable vs should-refuse questions."""
    inside = sorted(r["top_score"] for r in records if r.get("top_score") is not None and r.get("answerable"))
    outside = sorted(r["top_score"] for r in records if r.get("top_score") is not None and should_refuse(r))
    if not inside or not outside:
        return
    reranked = any(x.get("rerank_score") is not None for r in records for x in r.get("retrieved") or [])
    name, current = ("cross-encoder score", config.RERANK_MIN_SCORE) if reranked else ("cosine", config.MIN_SCORE)
    print(f"\nRefusal threshold (best {name} of each question)")
    print(f"  answerable    : min {inside[0]:.2f} · 10th pct {inside[len(inside) // 10]:.2f} · median {statistics.median(inside):.2f}")
    print(f"  should refuse : max {outside[-1]:.2f} · median {statistics.median(outside):.2f}")

    def counts(threshold):
        return sum(s < threshold for s in outside), sum(s < threshold for s in inside)

    correct, wrong = counts(current)
    print(f"  current {current}: correct refusals {correct}/{len(outside)}, wrong refusals {wrong}/{len(inside)}")
    scores = sorted(set(inside + outside))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])] or [current]
    best = max(candidates, key=lambda t: (counts(t)[0] - counts(t)[1], -abs(t - current)))
    correct, wrong = counts(best)
    print(f"  suggested {best:.2f}: correct refusals {correct}/{len(outside)}, wrong refusals {wrong}/{len(inside)}")


def agent_report(records: list[dict]):
    agents = [r for r in records if r.get("agent")]
    if not agents:
        return
    infos = [r["agent"] for r in agents]
    tools = Counter(t for a in infos for t in a["tools_used"])
    print("\nAgent trajectory (diagnosis, not grading)")
    print(f"  avg model calls {statistics.mean(a['steps'] for a in infos):.1f} · step-cap hits "
          f"{sum(a['hit_step_cap'] for a in infos)} · repeated identical calls {sum(a['repeated_calls'] for a in infos)} "
          f"· send attempts {sum(a['send_attempts'] for a in infos)} (send guard blocked {sum(a['send_guard_blocks'] for a in infos)})")
    print(f"  expected tools used: {pct([a.get('expected_tools_used') for a in infos])} · out-of-scope stopped cleanly: "
          f"{pct([a.get('stopped_cleanly') for a in infos])}")
    print(f"  tools used: {dict(tools)}")
    groups = defaultdict(list)
    for r in agents:
        groups[r.get("sub_type") if r.get("category") == "agent" else "priority question"].append(r)
    print(f"\n  {'scenario':<28} {'n':>3}  {'pass':<12} {'tools ok':<12} {'steps':>5} {'cap':>4} {'repeats':>7} {'latency':>8}")
    for name in sorted(groups, key=str):
        rows = groups[name]
        info = [r["agent"] for r in rows]
        print(f"  {str(name):<28} {len(rows):>3}  {pct([r['passed'] for r in rows]):<12} "
              f"{pct([a.get('expected_tools_used') for a in info]):<12} {mean_text([a['steps'] for a in info]):>5} "
              f"{sum(a['hit_step_cap'] for a in info):>4} {sum(a['repeated_calls'] for a in info):>7} "
              f"{mean_text([r.get('latency_s') for r in rows]):>7}s")


def print_report(records: list[dict], mode: str, runs: int, spent: float):
    first = [r for r in records if r["run"] == 0]
    print("\n" + "=" * 78)
    print(f"SCORECARD  mode={mode}  questions={len(first)}  runs={runs}  cost=${spent:.4f}")
    print("=" * 78)
    print(f"Retrieval  : {ranking_line(first)}")
    follow_ups = [r for r in first if r.get("history") and r.get("category") != "agent"]
    if follow_ups:
        print(f"Follow-ups : {ranking_line(follow_ups)} · pass {pct([r['passed'] for r in follow_ups])}")
    print(f"Pass rate  : {pct([r['passed'] for r in first])}")
    print(f"Refusals   : correct {pct([r['refused'] for r in first if should_refuse(r)])} · "
          f"wrong (answerable but refused) {sum(bool(r.get('refused')) for r in first if r.get('answerable'))}")
    answered = [r for r in first if r.get("answer") and not r.get("refused")]
    if answered:
        print(f"Citations  : {pct([r['has_citation'] for r in answered])} of answers cite a page")
    latencies = [r["latency_s"] for r in first if r.get("latency_s") is not None]
    if latencies:
        print(f"Latency    : avg {statistics.mean(latencies):.2f}s · median {statistics.median(latencies):.2f}s · max {max(latencies):.2f}s")
    print(f"Model calls: avg {mean_text([r.get('model_calls') for r in first])} per question · "
          f"cost ${spent / max(len(records), 1):.5f} per question (judge included)")
    unjudged = sum(r["outcome"] == "unjudged" for r in first)
    if unjudged:
        print(f"Unjudged   : {unjudged} (add --judge, or check the answers in data/eval_results.json by hand)")

    measured = [r["metrics"] for r in first if r.get("metrics") and "error" not in r["metrics"]]
    if measured:
        print(f"\nAnswer metrics (n={len(measured)}): faithfulness {pct([m['faithfulness'] for m in measured])} · "
              f"answer relevancy {statistics.mean(m['answer_relevancy'] for m in measured):.2f} · "
              f"context precision {statistics.mean(m['context_precision'] for m in measured):.2f}")

    group_table(first, "category", "By category")
    group_table(first, "difficulty", "By difficulty")
    group_table([r for r in first if r.get("sub_type") and r.get("category") != "agent"], "sub_type", "By sub_type")
    agent_report(first)
    if mode != "agent":
        calibration(first)

    if runs > 1:
        by_q = defaultdict(list)
        for r in records:
            by_q[r["id"]].append(r["passed"])
        judged = {qid: v for qid, v in by_q.items() if all(p is not None for p in v)}
        if judged:
            every = sum(all(v) for v in judged.values())
            plain = sum(sum(v) for v in judged.values()) / sum(len(v) for v in judged.values())
            flaky = [q for q, v in judged.items() if any(v) and not all(v)]
            print(f"\nReliability: plain pass rate {100 * plain:.0f}% · passes all {runs} runs {100 * every / len(judged):.0f}% "
                  f"· flaky: {', '.join(flaky) or 'none'}")

    priority = [r for r in first if r.get("priority")]
    if priority:
        print("\nPriority questions")
        for r in priority:
            print(f"  {ICON[r['outcome']]} {r['id']} [retrieval {mark((r.get('retrieval') or {}).get('hit'))}] {r['question'][:70]}")
    failures = [r for r in first if r["outcome"] in ("fail", "error")]
    if failures:
        print(f"\nFailures ({len(failures)}), debug with: python scripts/retrieval_checker.py --id <id>")
        for r in failures:
            print(f"  {r['id']} {r.get('category', ''):<15} {r.get('failure_reason') or '':<40} {r['question'][:55]}")


def run_ablation(questions: list[dict], k: int | None) -> list[dict]:
    """Retrieval-only runs, each row adding one improvement on top of the previous one."""
    rows = []
    for name, options in ABLATION:
        cost_before = load_costs()["total"]
        records = [run_one(q, "retrieval", False, k, search_options=options) for q in questions]
        main_set = [r for r in records if not r.get("history")]
        follow_ups = [r for r in records if r.get("history")]
        latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
        m, f = ranking_metrics([r.get("retrieval") for r in main_set]), ranking_metrics([r.get("retrieval") for r in follow_ups])
        refuse = [r for r in records if should_refuse(r)]
        rows.append({"version": name, **{key: m.get(key) for key in ("recall@1", "recall@3", "recall@5", "mrr")},
                     "follow_up_recall@5": f.get("recall@5"),
                     "correct_refusals": f"{sum(bool(r.get('refused')) for r in refuse)}/{len(refuse)}",
                     "wrong_refusals": sum(bool(r.get("refused")) for r in records if r.get("answerable")),
                     "latency_s": statistics.mean(latencies) if latencies else None,
                     "cost": load_costs()["total"] - cost_before, "options": options})
        print(f"  done: {name}")
    return rows


def ablation_markdown(rows: list[dict]) -> str:
    def num(value):
        return "-" if value is None else f"{value:.2f}"
    lines = ["| Version | Recall@1 | Recall@3 | Recall@5 | MRR | Follow-ups recall@5 | Correct refusals | Wrong refusals | Latency (s) |",
             "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    lines += [f"| {r['version']} | {num(r['recall@1'])} | {num(r['recall@3'])} | {num(r['recall@5'])} | {num(r['mrr'])} | "
              f"{num(r['follow_up_recall@5'])} | {r['correct_refusals']} | {r['wrong_refusals']} | {num(r['latency_s'])} |"
              for r in rows]
    return "\n".join(lines)


def compare(records: list[dict], previous_path: Path):
    old = {r["id"]: r for r in json.loads(previous_path.read_text(encoding="utf-8"))["results"] if r["run"] == 0}
    new = {r["id"]: r for r in records if r["run"] == 0}
    common = [i for i in new if i in old]
    regressions = [i for i in common if old[i]["passed"] is True and new[i]["passed"] is False]
    fixes = [i for i in common if old[i]["passed"] is False and new[i]["passed"] is True]
    lost_hits = [i for i in common if (old[i].get("retrieval") or {}).get("hit") is True
                 and (new[i].get("retrieval") or {}).get("hit") is False]
    print(f"\nRegression check vs {previous_path.name}")
    print(f"  fixed      ({len(fixes)}): {', '.join(fixes) or '-'}")
    print(f"  REGRESSED  ({len(regressions)}): {', '.join(regressions) or '-'}")
    print(f"  lost retrieval hits ({len(lost_hits)}): {', '.join(lost_hits) or '-'}")


def save(payload: dict, suffix: str) -> Path:
    config.EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    path = config.EVAL_RUNS_DIR / f"{payload['time']}_{suffix}.json"
    path.write_text(text, encoding="utf-8")
    if suffix != "ablation":
        config.EVAL_RESULTS_PATH.write_text(text, encoding="utf-8")
    return path


def settings(k: int | None) -> dict:
    return {"chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP, "retrieval_mode": config.RETRIEVAL_MODE,
            "top_k": k or config.TOP_K, "rerank": config.USE_RERANK, "rerank_model": config.RERANK_MODEL,
            "rerank_min_score": config.RERANK_MIN_SCORE, "min_score": config.MIN_SCORE,
            "query_rewrite": config.USE_QUERY_REWRITE, "query_expansion": config.QUERY_EXPANSION,
            "agent_max_steps": config.AGENT_MAX_STEPS, "agent_top_k": config.AGENT_TOP_K,
            "chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="rag", choices=["retrieval", "rag", "agent"])
    parser.add_argument("--set", choices=["rag", "agent", "all"],
                        help="questions to run (default: agent questions + priority for --mode agent, the rest otherwise)")
    parser.add_argument("--ablation", action="store_true", help="retrieval-only ablation table")
    parser.add_argument("--judge", action="store_true", help="grade answers with LLM-as-judge")
    parser.add_argument("--metrics", action="store_true", help="also faithfulness, answer relevancy, context precision")
    parser.add_argument("--runs", type=int, default=1, help="repeat each question (pass^k)")
    parser.add_argument("--k", type=int, help="passages kept per question (default TOP_K)")
    parser.add_argument("--category")
    parser.add_argument("--ids", help="comma-separated ids, e.g. q01,q07")
    parser.add_argument("--priority", action="store_true", help="only the 5 priority questions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--compare", type=Path, help="a previous data/eval_runs/*.json, for the regression check")
    parser.add_argument("--questions", type=Path, default=config.QUESTIONS_PATH)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    has_agent_questions = any(q.get("category") == "agent" for q in questions)
    set_name = "ids" if args.ids else args.set or ("agent" if args.mode == "agent" else "rag")
    if not args.ids:
        questions = question_set(questions, set_name)
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.priority:
        questions = [q for q in questions if q.get("priority")]
    if args.shuffle:
        random.shuffle(questions)
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit("No questions selected.")
    if args.mode == "agent" and not has_agent_questions:
        print("⚠️  No agent questions yet: running the priority questions only. Add your demo scenario questions "
              "to questions.json with \"category\": \"agent\" (see data/agent_questions_template.json).")

    if config.INJECTION_TEST:
        print("⚠️  INJECTION_TEST is on: the poisoned test passage is in the index. Turn it off for normal evals.")
    retrieve("warm up", rerank=args.ablation or config.USE_RERANK)  # load the models before anything is timed
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.ablation:
        print(f"Ablation on {len(questions)} questions (retrieval only)")
        rows = run_ablation(questions, args.k)
        table = ablation_markdown(rows)
        print("\n" + table)
        (config.DATA_DIR / "ablation.md").write_text(table + "\n", encoding="utf-8")
        path = save({"mode": "ablation", "time": stamp, "settings": settings(args.k), "rows": rows}, "ablation")
        print(f"\nSaved data/ablation.md and {path.relative_to(config.ROOT)}")
        return

    cost_before = load_costs()["total"]
    records = []
    for run in range(args.runs):
        for i, q in enumerate(questions, start=1):
            record = run_one(q, args.mode, args.judge, args.k, args.metrics)
            record["run"] = run
            records.append(record)
            print(f"[{run + 1}/{args.runs}] {i:>3}/{len(questions)} {ICON[record['outcome']]} {q['id']} "
                  f"retrieval {mark((record.get('retrieval') or {}).get('hit'))} {q.get('category', ''):<15} {q['question'][:55]}")
    spent = load_costs()["total"] - cost_before
    print_report(records, args.mode, args.runs, spent)
    path = save({"mode": args.mode, "judge": args.judge, "metrics": args.metrics, "runs": args.runs,
                 "questions_file": args.questions.name, "question_set": set_name, "time": stamp, "cost": spent,
                 "settings": settings(args.k), "results": records}, args.mode)
    print(f"\nSaved {config.EVAL_RESULTS_PATH.relative_to(config.ROOT)} and {path.relative_to(config.ROOT)}")
    if args.compare:
        compare(records, args.compare)


if __name__ == "__main__":
    main()
