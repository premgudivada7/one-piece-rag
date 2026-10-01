import json
import re
import statistics
from datetime import date
from pathlib import Path

from app import config
from app.llm import chat
from app.prompts import JUDGE_AMBIGUOUS_PROMPT, JUDGE_PROMPT, JUDGE_SCHEMA, METRICS_PROMPT, METRICS_SCHEMA
from app.rag import is_refusal

CATEGORIES = ["factual", "cross-reference", "stress-test", "out-of-scope", "ambiguous", "agent"]
CROSS_SUBTYPES = ["adjacent_chunks", "same_document", "different_documents", "repeated", "contradictory",
                  "summary_vs_detail"]
STRESS_SUBTYPES = ["paraphrased", "exact_term", "yes_no", "temporal", "negation", "multi_hop", "table_list",
                   "edge_of_chunk", "language_variation", "on_topic_unanswerable", "multi_part"]
# The agent demo scenarios; each agent eval question uses one of them as its sub_type.
AGENT_SCENARIOS = ["one_tool", "same_tool_several_times", "rephrase_and_retry", "search_then_calculate",
                   "own_tool", "no_tool", "act_with_approval", "out_of_scope", "injection_against_actions"]
RECALL_KS = (1, 3, 5)
_DAYS_SINCE = re.compile(r"\{days_since:(\d{4}-\d{2}-\d{2})\}")


def fill_dates(text):
    """An expected answer like "{days_since:2026-08-02} days" becomes "60 days" on 1 October 2026,
    so date questions don't go stale the next day."""
    if not isinstance(text, str):
        return text
    return _DAYS_SINCE.sub(lambda m: str((date.today() - date.fromisoformat(m.group(1))).days), text)


def load_questions(path: Path = config.QUESTIONS_PATH) -> list[dict]:
    """The eval set, each question with an id (q01, q02...) when it has none."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put your Assignment 1 questions.json in data/.")
    questions = json.loads(path.read_text(encoding="utf-8"))
    for i, q in enumerate(questions, start=1):
        q.setdefault("id", f"q{i:02d}")
        q["expected_answer"] = fill_dates(q.get("expected_answer"))
    return questions


def question_set(questions: list[dict], name: str) -> list[dict]:
    """"rag": everything except the agent questions. "agent": the agent questions plus the
    5 priority questions, as the project asks. "all": everything."""
    if name == "rag":
        return [q for q in questions if q.get("category") != "agent"]
    if name == "agent":
        return [q for q in questions if q.get("category") == "agent" or q.get("priority")]
    return questions


def should_refuse(q: dict) -> bool:
    """The right behaviour is to say the documents don't cover it: an out-of-scope or
    on-topic-unanswerable question, or one whose expected answer is itself a refusal."""
    return (q.get("category") == "out-of-scope" or q.get("sub_type") in ("on_topic_unanswerable", "out_of_scope")
            or is_refusal(q.get("expected_answer") or ""))


def _norm_source(name: str) -> str:
    return Path(str(name)).name.lower().strip()


def expected_pairs(q: dict) -> list[tuple[str, int]]:
    """(source, page) pairs from expected_chunks, or from source/page when there are none."""
    chunks = q.get("expected_chunks")
    if chunks:
        return [(_norm_source(c["source"]), int(c["page"])) for c in chunks if c.get("source") and c.get("page")]
    if q.get("source") and q.get("page") is not None:
        pages = q["page"] if isinstance(q["page"], list) else [q["page"]]
        return [(_norm_source(q["source"]), int(p)) for p in pages]
    return []


def retrieval_check(q: dict, results: list[dict]) -> dict:
    """Which expected pages came back, and the rank of the first chunk on an expected page.
    A chunk covering pages [4, 5] counts for an expected page 4 or 5."""
    expected = expected_pairs(q)
    if not expected:
        return {"expected": [], "found": [], "missing": [], "hit": None, "recall": None, "first_rank": None}
    found, first_rank = [], None
    for pair in expected:
        for rank, r in enumerate(results, start=1):
            if _norm_source(r["source"]) == pair[0] and pair[1] in (r.get("pages") or [r["page"]]):
                found.append(pair)
                first_rank = rank if first_rank is None else min(first_rank, rank)
                break
    missing = [p for p in expected if p not in found]
    return {"expected": expected, "found": found, "missing": missing, "hit": not missing,
            "recall": len(found) / len(expected), "first_rank": first_rank}


def ranking_metrics(checks: list[dict]) -> dict:
    """recall@1/3/5 (share of questions with an expected chunk in the top k) and MRR (mean of
    1 / rank of the first expected chunk, 0 when none came back), on questions with expected chunks."""
    checks = [c for c in checks if c and c.get("expected")]
    if not checks:
        return {"n": 0}
    ranks = [c["first_rank"] for c in checks]
    out = {"n": len(checks), "mrr": statistics.mean(1 / r if r else 0.0 for r in ranks),
           "full_recall": statistics.mean(1.0 if c["hit"] else 0.0 for c in checks)}
    for k in RECALL_KS:
        out[f"recall@{k}"] = statistics.mean(1.0 if r and r <= k else 0.0 for r in ranks)
    return out


def pct(values) -> str:
    values = [v for v in values if v is not None]
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "-"


def mean_text(values, fmt="{:.1f}") -> str:
    values = [v for v in values if v is not None]
    return fmt.format(statistics.mean(values)) if values else "-"


def answer_overlap(expected: str, answer: str) -> float | None:
    """A free rough signal (no API call): share of the expected answer's numbers and longer words found in the answer."""
    if not expected:
        return None
    keys = set(re.findall(r"\d+(?:[.,]\d+)?|[^\W\d_]{4,}", expected.lower()))
    if not keys:
        return None
    answer = (answer or "").lower()
    return round(sum(k in answer for k in keys) / len(keys), 2)


def _json_call(system: str, user: str, operation: str, schema: dict, max_tokens: int) -> dict:
    response = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    operation=operation, max_tokens=max_tokens, response_format=schema)
    return json.loads(response.choices[0].message.content)


def judge(q: dict, answer: str) -> dict:
    """LLM-as-judge with a strict JSON answer. A first pass only: we check answers by hand."""
    if q.get("category") == "ambiguous":
        system, user = JUDGE_AMBIGUOUS_PROMPT, f"QUESTION: {q['question']}\n\nANSWER: {answer}"
    else:
        system = JUDGE_PROMPT
        user = f"QUESTION: {q['question']}\n\nEXPECTED: {q.get('expected_answer')}\n\nANSWER: {answer}"
    try:
        return _json_call(system, user, "judge", JUDGE_SCHEMA, max_tokens=400)
    except (TypeError, json.JSONDecodeError):
        return {"verdict": "unclear", "reason": "judge returned invalid JSON"}


METRICS_MAX_PASSAGES = 12
METRICS_PASSAGE_CHARS = 1200


def rag_metrics(question: str, answer: str, passages: list[dict]) -> dict | None:
    """Faithfulness, answer relevancy and context precision from one judge call
    (context recall needs no model: see retrieval_check)."""
    passages = [p for p in passages if p.get("text")][:METRICS_MAX_PASSAGES]
    if not answer or not passages:
        return None
    numbered = "\n\n".join(f"[{i}] ({p['source']} p.{p['page']})\n{p['text'][:METRICS_PASSAGE_CHARS]}"
                           for i, p in enumerate(passages, start=1))
    try:
        verdict = _json_call(METRICS_PROMPT, f"QUESTION: {question}\n\nPASSAGES:\n{numbered}\n\nANSWER: {answer}",
                             "judge_metrics", METRICS_SCHEMA, max_tokens=300)
    except Exception as e:  # a metrics failure must not stop the eval run
        return {"error": f"{type(e).__name__}: {e}"}
    relevant = {i for i in verdict["relevant_passages"] if 1 <= i <= len(passages)}
    return {"faithfulness": bool(verdict["faithful"]), "unsupported_claims": verdict["unsupported_claims"],
            "answer_relevancy": {"yes": 1.0, "partly": 0.5, "no": 0.0}[verdict["answers_question"]],
            "context_precision": round(len(relevant) / len(passages), 3), "relevant_passages": sorted(relevant),
            "passages_judged": len(passages)}
