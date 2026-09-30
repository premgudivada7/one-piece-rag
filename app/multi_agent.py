
import json
import sys
import time
from datetime import datetime

from agent import (
    CALCULATOR_TOOL,
    DATE_TOOL,
    GET_ARTICLE_TOOL,
    FALLBACK_MESSAGE,
    MIN_SCORE,
    find_citations,
    run_agent,
    save_trajectory,
)
from calculator import calculator
from date_tool import date_tool
from get_article import get_article
from quote_exact import quote_exact
from send_message import SEND_MESSAGE_TOOL, propose_message
from term_counter import count_term
from retrieval import load_data, retrieve


# ---------- Configuration ----------

COORDINATOR_MAX_STEPS = 4
SPECIALIST_MAX_STEPS = 5

DEFAULT_K = 5
MAX_K = 10


DOCUMENTS = {
    "eu": {
        "source": "eu_ai_act_2024.pdf",
        "title": "EU Artificial Intelligence Act (Regulation (EU) 2024/1689)",
        "agent_name": "eu_ai_act_agent",
        "tool_name": "ask_eu_ai_act_agent",
    },
    "nist": {
        "source": "nist_ai_rmf_1_0.pdf",
        "title": "NIST AI Risk Management Framework 1.0",
        "agent_name": "nist_ai_rmf_agent",
        "tool_name": "ask_nist_ai_rmf_agent",
    },
}


# ---------- Specialist tools (limited to one document) ----------

def make_search(source: str):
    """Create a search tool that only returns passages from one document."""

    def search_document(query: str, k: int = DEFAULT_K) -> dict:
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "Provide a non-empty search query."}

        if type(k) is not int or not 1 <= k <= MAX_K:
            return {"ok": False, "error": f"k must be an integer from 1 to {MAX_K}."}

        try:
            # Rank every chunk (one embedding call), keep this document's.
            total_chunks = len(load_data()[0])
            pool = retrieve(query.strip(), k=total_chunks)
        except Exception as error:
            return {"ok": False, "error": f"Search failed: {error}"}

        chunks = [c for c in pool if c["source"] == source][:k]

        if not chunks or max(c["score"] for c in chunks) < MIN_SCORE:
            return {
                "ok": False,
                "query": query.strip(),
                "error": f"No relevant passages found in {source}.",
                "hint": "Rephrase with other keywords, or report that "
                        "this document does not cover it.",
            }

        return {
            "ok": True,
            "query": query.strip(),
            "results": [
                {
                    "ref": number,
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": round(chunk["score"], 4),
                    "text": chunk["text"],
                }
                for number, chunk in enumerate(chunks, start=1)
            ],
        }

    return search_document


def make_quote(source: str):
    """Create a quote tool fixed to one document."""

    def quote_page(page: int) -> dict:
        return quote_exact(source, page)

    return quote_page


def make_count(source: str):
    """Create a count_term tool fixed to one document."""

    def count_in_document(term: str, max_pages: int = 10) -> dict:
        return count_term(term, source=source, max_pages=max_pages)

    return count_in_document


def specialist_tools(source: str) -> tuple[list, dict]:
    """Tool schemas and functions for one specialist agent."""
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": (
                    f"Search {source} for passages relevant to a query. "
                    "Returns passages with page numbers and similarity "
                    "scores. If ok=false, rephrase and search again."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "A focused search query."},
                        "k": {"type": "integer",
                              "description": f"Passages to return (1 to {MAX_K})."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "quote_exact",
                "description": (
                    f"Return the full text of one page of {source}, to quote "
                    "or verify a passage (for example which Article it is in)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer",
                                 "description": "PDF page number, starting at 1."},
                    },
                    "required": ["page"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "count_term",
                "description": (
                    f"Count exact occurrences of a word or phrase in {source} "
                    "(case-insensitive, whole term), with the pages that "
                    "mention it most. Use it for 'how many times' questions "
                    "and to locate exact terms like 'Article 99'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string",
                                 "description": "The exact word or phrase."},
                        "max_pages": {"type": "integer",
                                      "description": "Top pages to list (1 to 50)."},
                    },
                    "required": ["term"],
                    "additionalProperties": False,
                },
            },
        },
        CALCULATOR_TOOL,
    ]

    functions = {
        "search_corpus": make_search(source),
        "quote_exact": make_quote(source),
        "count_term": make_count(source),
        "calculator": calculator,
    }

    # Only the EU AI Act has numbered Articles.
    if source == DOCUMENTS["eu"]["source"]:
        schemas.append(GET_ARTICLE_TOOL)
        functions["get_article"] = get_article

    return schemas, functions


def specialist_prompt(document: dict) -> str:
    """Instructions for one specialist agent."""
    return f"""
You are the {document['title']} specialist. You can only see
{document['source']}. Another agent handles the other document.

Rules:
1. Search your document before answering. For several parts, search each.
2. If a search returns ok=false or irrelevant passages, rephrase and try
   again (at most 2 retries): use the distinctive words of the expected
   answer, leave out the document name, and ask for k=10.
3. Use ONLY the tool results. No outside knowledge.
4. When the question names an Article or section, or passages give
   different values, confirm which Article or section each value comes
   from: use get_article for EU AI Act Articles (if you have it), and
   quote_exact for pages.
5. Use count_term for "how many times / how many pages" questions and to
   find exact terms (article numbers, category IDs). Use the calculator
   for any arithmetic.
6. Cite every factual claim as ({document['source']}, p. N).
7. If your document does not contain the answer, reply exactly:
   "{document['source']} does not contain information about this."
8. Be concise: give the facts the coordinator needs, with citations.
   Every sentence must be supported by a tool result: add no explanations
   or descriptions that the passages do not contain.
""".strip()


# ---------- Coordinator ----------

COORDINATOR_PROMPT = f"""
You coordinate two specialist agents that answer questions about AI
regulation documents:
- ask_eu_ai_act_agent: knows only the EU AI Act (eu_ai_act_2024.pdf)
- ask_nist_ai_rmf_agent: knows only the NIST AI RMF 1.0 (nist_ai_rmf_1_0.pdf)

How to work:
1. Question about one document: ask that specialist.
2. Question that compares, links or involves both documents, or where you
   are unsure which document covers it: ask BOTH specialists, in the same
   step, each with a focused question about its own document.
3. Use the calculator for arithmetic on numbers the specialists return,
   and date_tool for "how long until / since" a date (never count days
   yourself). Ask a specialist for the date first, wait for its answer,
   then call date_tool: never use dates from memory or placeholders.
4. Combine the specialists' answers into one clear answer. Keep their
   citations exactly as written, e.g. (eu_ai_act_2024.pdf, p. 56).
   Do not add facts or citations that the specialists did not give.
5. If the specialists find nothing relevant, reply exactly:
   "{FALLBACK_MESSAGE}"
6. Greetings or questions about what you can do: answer briefly without tools.
7. Remember the documents have different legal status: the EU AI Act is
   binding law, the NIST AI RMF is voluntary guidance.
8. send_message: ONLY if the user explicitly asks to send/post/share the
   answer. Call it once with the final combined answer and citations; it
   only proposes the message, a person must approve it.
""".strip()


def specialist_tool_schema(document: dict) -> dict:
    """The coordinator sees each specialist as one tool."""
    return {
        "type": "function",
        "function": {
            "name": document["tool_name"],
            "description": (
                f"Ask the {document['title']} specialist agent a question. "
                f"It searches only {document['source']} and returns a cited "
                "answer, or says the document does not cover it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A focused question about this document.",
                    },
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    }


def make_specialist(document: dict, record: list):
    """Create a coordinator tool that runs one specialist agent."""
    schemas, functions = specialist_tools(document["source"])
    prompt = specialist_prompt(document)

    def ask_specialist(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "Provide a non-empty question."}

        run = run_agent(
            question,
            max_steps=SPECIALIST_MAX_STEPS,
            system_prompt=prompt,
            tools=schemas,
            tool_functions=functions,
            agent_name=document["agent_name"],
            save=False,
        )
        record.append(run)

        tools_used = [step["tool"] for step in run["steps"]]
        return {
            "ok": True,
            "agent": document["agent_name"],
            "answer": run["answer"],
            "citations": run["citations"],
            "summary": (f"{len(run['citations'])} citations, "
                        f"tools: {', '.join(tools_used) or 'none'}"),
        }

    return ask_specialist


def run_multi_agent(question: str) -> dict:
    """Answer a question with the coordinator and the two specialists."""
    started = time.time()
    specialist_runs: list[dict] = []

    tools = [specialist_tool_schema(d) for d in DOCUMENTS.values()]
    tools += [CALCULATOR_TOOL, DATE_TOOL, SEND_MESSAGE_TOOL]

    functions = {d["tool_name"]: make_specialist(d, specialist_runs)
                 for d in DOCUMENTS.values()}
    functions["calculator"] = calculator
    functions["date_tool"] = date_tool
    functions["send_message"] = propose_message   # only proposes

    coordinator = run_agent(
        question,
        max_steps=COORDINATOR_MAX_STEPS,
        system_prompt=COORDINATOR_PROMPT,
        tools=tools,
        tool_functions=functions,
        agent_name="coordinator",
        save=False,
    )

    run = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "agent": "multi_agent",
        "question": question,
        "answer": coordinator["answer"],
        "citations": find_citations(coordinator["answer"] or ""),
        "stop_reason": coordinator["stop_reason"],
        "agents_used": sorted({r["agent"] for r in specialist_runs}),
        "coordinator_steps": coordinator["steps"],
        "specialist_runs": specialist_runs,
        "pending_messages": coordinator["pending_messages"],
        "model_calls": coordinator["model_calls"]
                       + sum(r["model_calls"] for r in specialist_runs),
        "seconds": round(time.time() - started, 2),
    }
    save_trajectory(run)
    return run


# ---------- Run from the terminal ----------

def print_multi_run(run: dict) -> None:
    """Print which agents worked, their tool calls, and the final answer."""
    print(f"\nQuestion: {run['question']}\n")
    print(f"Agents used: {', '.join(run['agents_used']) or 'coordinator only'}\n")

    print("Coordinator:")
    if not run["coordinator_steps"]:
        print("  (answered directly)")
    for step in run["coordinator_steps"]:
        mark = "✅" if step["ok"] else "❌"
        arguments = step["arguments"]
        if isinstance(arguments, dict) and "question" in arguments:
            detail = arguments["question"]
        else:
            detail = f"{json.dumps(arguments, ensure_ascii=False)} → {step['summary']}"
        print(f"  step {step['step']} {mark} {step['tool']}: {detail}")

    for specialist in run["specialist_runs"]:
        print(f"\n{specialist['agent']}  ← \"{specialist['question']}\"")
        for step in specialist["steps"]:
            mark = "✅" if step["ok"] else "❌"
            print(f"  step {step['step']} {mark} {step['tool']}"
                  f"({json.dumps(step['arguments'], ensure_ascii=False)})")
            print(f"         → {step['summary']}")
        print(f"  answer: {specialist['answer']}")

    print(f"\nFinal answer:\n{run['answer']}\n")
    print(f"Citations: {run['citations'] or 'none'}")
    print(f"Model calls: {run['model_calls']} | Stop: {run['stop_reason']}"
          f" | {run['seconds']}s")


def main() -> None:
    question = " ".join(sys.argv[1:]) or (
        "Do the EU AI Act and the NIST AI RMF both say risk management "
        "must continue across the AI lifecycle?"
    )
    print_multi_run(run_multi_agent(question))


if __name__ == "__main__":
    main()
