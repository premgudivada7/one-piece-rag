"""Agentic AI assistant: the model decides which tools to use.

Tools (all read-only):
    search_corpus  find passages in the EU AI Act / NIST AI RMF
    calculator     exact arithmetic (Member 2's calculator.py)
    quote_exact    full text of one document page (Member 2's quote_exact.py)
    count_term     exact counts/pages of a word or phrase (term_counter.py)
    get_article    full text of an EU AI Act Article by number (get_article.py)
    date_tool      today's date and days between two dates (date_tool.py)

The agent loop (think -> act -> observe -> repeat):
    1. Send the question and the tool list to gpt-4o-mini.
    2. If the model asks for tools, run them (several can run in the same
       step, e.g. search_corpus + calculator), give it the results, repeat.
    3. When the model answers without asking for a tool, stop.
    4. A hard step cap stops runaway loops.
Every tool call is recorded in a trajectory and saved to
data/trajectories.jsonl so runs can be inspected.

Terminal:
    python app/agent.py "What is the maximum fine under Article 99 for prohibited AI practices, and what is half of it?"
    python app/agent.py --search "NIST core functions"   # search tool only
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from calculator import calculator
from cost_tracker import track_cost
from date_tool import DATE_TOOL, date_tool
from get_article import GET_ARTICLE_TOOL, get_article
from quote_exact import quote_exact
from retrieval import get_client, retrieve
from term_counter import COUNT_TERM_TOOL, count_term


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAJECTORY_FILE = PROJECT_ROOT / "data" / "trajectories.jsonl"

MODEL = "gpt-4o-mini"
MAX_STEPS = 6            # model calls per question, including the final answer

DEFAULT_K = 5
MAX_K = 10

# Below this similarity the passages are almost certainly unrelated
# (out-of-scope questions usually score around 0.1 to 0.2).
MIN_SCORE = 0.25

SOURCES = ["eu_ai_act_2024.pdf", "nist_ai_rmf_1_0.pdf"]

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided "
    "documents to answer this question."
)

SYSTEM_PROMPT = f"""
You are the One Piece AI Regulation assistant. You answer questions about
two documents only: the EU AI Act (eu_ai_act_2024.pdf) and the NIST AI Risk
Management Framework 1.0 (nist_ai_rmf_1_0.pdf).

Tools:
- search_corpus: find relevant passages. Use it for every question about
  the documents. For questions with several parts, search each part.
- calculator: use it for ANY arithmetic (counts, sums, differences,
  percentages, dates as numbers). Never calculate in your head.
- quote_exact: get the full text of one page when you must quote exactly,
  or when a search result is cut off and you need the rest of the page.
- count_term: exact count of a word or phrase per document and page. Use it
  for "how many times / how many pages" questions and to find exact terms
  such as "Article 99" or "GOVERN 1.2" when search misses them.
- get_article: exact, complete text of an EU AI Act Article by number,
  optionally one paragraph (Article 99(3) -> article=99, paragraph=3) or
  one definition point (Article 3 point (4) -> article=3, paragraph=4).
- date_tool: today's date, or the number of days between two dates. Use it
  for "how long until / since" questions; never count days yourself.

You may call several tools in the same step ONLY when they do not depend on
each other (for example two searches). If a tool needs a value that another
tool must find first (a fine to halve, a date to count from), wait for that
result and call the next tool in a later step. Never put placeholders
(like "fine_value") or guessed values in tool arguments.

Rules:
1. Use ONLY information returned by the tools. No outside knowledge.
2. If search_corpus returns ok=false or irrelevant passages, rephrase the
   query with different keywords and search again (at most 2 retries).
3. If the documents still do not contain the answer, reply exactly:
   "{FALLBACK_MESSAGE}"
4. Greetings or questions about what you can do: answer briefly without tools.
5. Cite every factual claim as (filename, p. N) using the source and page
   from the tool results, e.g. (eu_ai_act_2024.pdf, p. 46). Never invent
   sources or pages.
6. Keep the answer clear and concise.
7. Search passages are short excerpts and often do not show which Article
   or section they belong to. When the question names an EU AI Act
   Article, read it with get_article. When passages give different values
   (for example different fines or dates), use get_article or quote_exact
   to confirm which Article each value comes from before answering.
   Answer from the Article the question asks about. Cite get_article
   results with the page(s) it returns.
8. Every date you pass to date_tool (other than "today") must come from a
   tool result in this conversation, never from memory. For EU AI Act
   application dates, read Article 113 with get_article first. Cite the
   page the date comes from.
""".strip()


# ---------- Tool 1: search_corpus ----------

def search_corpus(query: str, k: int = DEFAULT_K) -> dict:
    """Search the EU AI Act and NIST AI RMF for passages about a query."""
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "Provide a non-empty search query."}

    if type(k) is not int or not 1 <= k <= MAX_K:
        return {"ok": False, "error": f"k must be an integer from 1 to {MAX_K}."}

    query = query.strip()

    try:
        chunks = retrieve(query, k=k)
    except Exception as error:  # API or file problems become tool errors.
        return {"ok": False, "error": f"Search failed: {error}"}

    if not chunks:
        return {
            "ok": False,
            "query": query,
            "error": "No passages were found.",
            "hint": "Rephrase the query with different keywords.",
        }

    best_score = chunks[0]["score"]

    if best_score < MIN_SCORE:
        return {
            "ok": False,
            "query": query,
            "best_score": round(best_score, 4),
            "error": "No relevant passages: the documents may not cover this.",
            "hint": (
                "Rephrase with terms used in the EU AI Act or NIST AI RMF, "
                "or answer that the documents do not contain this."
            ),
        }

    results = [
        {
            "ref": number,
            "source": chunk["source"],
            "page": chunk["page"],
            "score": round(chunk["score"], 4),
            "text": chunk["text"],
        }
        for number, chunk in enumerate(chunks, start=1)
    ]

    return {"ok": True, "query": query, "results": results}


# ---------- Tool schemas the model sees ----------

SEARCH_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the EU AI Act (eu_ai_act_2024.pdf) and the NIST AI Risk "
            "Management Framework 1.0 (nist_ai_rmf_1_0.pdf) for passages "
            "relevant to a query. Returns passages with their source file, "
            "page number and similarity score. If it returns ok=false, "
            "rephrase the query with different keywords and search again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A focused search query, e.g. 'Article 5 prohibited "
                        "AI practices' or 'NIST GOVERN function'."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": (
                        f"Number of passages to return (1 to {MAX_K}, "
                        f"default {DEFAULT_K})."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression exactly. Supports + - * / "
            "// % and parentheses with plain numbers (no commas, no units, "
            "no exponent). Use it for every calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "For example '(144 + 48) / 2'.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

QUOTE_EXACT_TOOL = {
    "type": "function",
    "function": {
        "name": "quote_exact",
        "description": (
            "Return the complete stored text of one page of a document. Use "
            "it to quote or verify a passage that search_corpus pointed to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": SOURCES,
                    "description": "Exact document filename.",
                },
                "page": {
                    "type": "integer",
                    "description": "PDF page number, starting at 1.",
                },
            },
            "required": ["source", "page"],
            "additionalProperties": False,
        },
    },
}


# ---------- Tool registry ----------

TOOLS = [
    SEARCH_CORPUS_TOOL,
    CALCULATOR_TOOL,
    QUOTE_EXACT_TOOL,
    COUNT_TERM_TOOL,
    GET_ARTICLE_TOOL,
    DATE_TOOL,
]

TOOL_FUNCTIONS = {
    "search_corpus": search_corpus,
    "calculator": calculator,
    "quote_exact": quote_exact,
    "count_term": count_term,
    "get_article": get_article,
    "date_tool": date_tool,
}


def run_tool(name: str, arguments: str | dict,
             tool_functions: dict | None = None) -> dict:
    """Run one tool call from the model and always return a dict."""
    tool_functions = tool_functions or TOOL_FUNCTIONS
    function = tool_functions.get(name)

    if function is None:
        return {
            "ok": False,
            "error": f"Unknown tool: {name}",
            "available_tools": sorted(tool_functions),
        }

    # The model sends arguments as a JSON string.
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "Tool arguments are not valid JSON."}

    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object."}

    try:
        return function(**arguments)
    except TypeError as error:  # Missing or unexpected argument names.
        return {"ok": False, "error": f"Invalid arguments for {name}: {error}"}


# ---------- Trajectory helpers ----------

def summarize_result(name: str, result: dict) -> str:
    """A short, human-readable line describing a tool result."""
    if not result.get("ok"):
        return f"failed: {result.get('error', 'unknown error')}"

    if name == "search_corpus":
        pages = ", ".join(
            f"{r['source'].split('_')[0].upper()} p.{r['page']}"
            for r in result["results"]
        )
        return f"{len(result['results'])} passages ({pages})"

    if name == "calculator":
        return f"{result['expression']} = {result['result']}"

    if name == "quote_exact":
        return f"{result['source']} p.{result['page']} ({result['char_count']} chars)"

    if name == "count_term":
        return (f"'{result['term']}': {result['total_count']} times on "
                f"{result['pages_with_term']} pages {result['by_document']}")

    if name == "get_article":
        return (f"{result['reference']} '{result['title']}' pages {result['pages']}"
                f" (part {result['part']}/{result['parts']})")

    if name == "date_tool":
        if result["operation"] == "today":
            return f"today is {result['today']}"
        return f"{result['start']} → {result['end']} = {result['days']} days"

    # Other tools (e.g. the specialist agents) may provide their own summary.
    return result.get("summary", "ok")


def find_citations(answer: str) -> list[dict]:
    """Collect (filename, p. N) citations written in the answer."""
    found = []
    pattern = r"\(?\s*([\w.-]+\.pdf)\s*,\s*p(?:age|\.)?\s*(\d+)"

    for source, page in re.findall(pattern, answer):
        citation = {"source": source, "page": int(page)}
        if citation not in found:
            found.append(citation)

    return found


def save_trajectory(run: dict) -> None:
    """Append one run to data/trajectories.jsonl (one JSON object per line)."""
    TRAJECTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRAJECTORY_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(run, ensure_ascii=False) + "\n")


# ---------- The agent loop ----------

def run_agent(
    question: str,
    max_steps: int = MAX_STEPS,
    system_prompt: str = SYSTEM_PROMPT,
    tools: list | None = None,
    tool_functions: dict | None = None,
    agent_name: str = "agent",
    save: bool = True,
) -> dict:
    """Answer a question by letting the model call tools in a loop.

    The optional settings let other agents (see multi_agent.py) reuse
    this same loop with their own instructions and tools.
    """
    tools = tools or TOOLS
    tool_functions = tool_functions or TOOL_FUNCTIONS
    started = time.time()
    run = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "agent": agent_name,
        "question": question,
        "steps": [],
        "answer": None,
        "citations": [],
        "stop_reason": None,
        "model_calls": 0,
    }

    if not isinstance(question, str) or not question.strip():
        run["answer"] = "Please enter a question."
        run["stop_reason"] = "empty_question"
        return run

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
    ]
    client = get_client()

    for step in range(1, max_steps + 1):
        # On the last allowed step, the model must answer (no more tools).
        last_step = step == max_steps

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=messages,
            tools=tools,
            tool_choice="none" if last_step else "auto",
        )
        track_cost(response)
        run["model_calls"] += 1

        message = response.choices[0].message

        # No tool calls: this is the final answer.
        if not message.tool_calls:
            run["answer"] = (message.content or "").strip() or FALLBACK_MESSAGE
            run["stop_reason"] = "step_cap" if last_step else "answered"
            break

        # Keep the model's tool request in the conversation.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ],
        })

        # Run every tool the model asked for in this step.
        for call in message.tool_calls:
            name = call.function.name
            result = run_tool(name, call.function.arguments, tool_functions)

            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = call.function.arguments

            run["steps"].append({
                "step": step,
                "tool": name,
                "arguments": arguments,
                "ok": bool(result.get("ok")),
                "summary": summarize_result(name, result),
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    run["citations"] = find_citations(run["answer"] or "")
    run["seconds"] = round(time.time() - started, 2)
    if save:
        save_trajectory(run)
    return run


# ---------- Run from the terminal ----------

def print_run(run: dict) -> None:
    """Print the trajectory and the final answer."""
    print(f"\nQuestion: {run['question']}\n")
    print("Trajectory:")
    if not run["steps"]:
        print("  (no tools used)")
    for item in run["steps"]:
        mark = "✅" if item["ok"] else "❌"
        print(f"  step {item['step']} {mark} {item['tool']}"
              f"({json.dumps(item['arguments'], ensure_ascii=False)})")
        print(f"         → {item['summary']}")

    print(f"\nAnswer:\n{run['answer']}\n")
    print(f"Citations: {run['citations'] or 'none'}")
    print(f"Model calls: {run['model_calls']} | Stop: {run['stop_reason']}"
          f" | {run.get('seconds', 0)}s")


def main() -> None:
    """Run the agent (or only the search tool with --search)."""
    args = sys.argv[1:]

    if args and args[0] == "--search":
        query = " ".join(args[1:]) or "What is a high-risk AI system?"
        print(json.dumps(search_corpus(query), ensure_ascii=False, indent=2))
        return

    question = " ".join(args) or (
        "What is the maximum fine under Article 99 of the EU AI Act for "
        "prohibited AI practices, and what is half of that amount?"
    )
    print_run(run_agent(question))


if __name__ == "__main__":
    main()
