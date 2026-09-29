import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import retrieve


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_K = 5
MAX_K = 10
MAX_SEARCH_ATTEMPTS = 3
MIN_SCORE = 0.25


# =========================================================
# OPENAI SETUP
# =========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# SEARCH CORPUS
# =========================================================

def search_corpus(query: str, k: int = DEFAULT_K) -> dict:

    if not isinstance(query, str) or not query.strip():
        return {
            "ok": False,
            "error": "Provide a non-empty search query."
        }

    if type(k) is not int or not 1 <= k <= MAX_K:
        return {
            "ok": False,
            "error": f"k must be an integer from 1 to {MAX_K}."
        }

    query = query.strip()

    print("\n[TOOL] search_corpus")
    print("[QUERY]", query)
    print("[K]", k)

    try:
        chunks = retrieve(query, k=k)

    except Exception as error:
        return {
            "ok": False,
            "error": f"Search failed: {error}"
        }

    if not chunks:
        return {
            "ok": False,
            "query": query,
            "error": "No passages were found.",
            "hint": "Try substantially different document terminology."
        }

    best_score = chunks[0]["score"]

    if best_score < MIN_SCORE:
        return {
            "ok": False,
            "query": query,
            "best_score": round(best_score, 4),
            "error": "No sufficiently relevant passages were found.",
            "hint": (
                "Try named concepts, headings, abbreviations, "
                "or terminology used in the source document."
            ),
        }

    results = []

    for number, chunk in enumerate(chunks, start=1):

        results.append(
            {
                "ref": number,
                "source": chunk["source"],
                "page": chunk["page"],
                "score": round(chunk["score"], 4),
                "text": chunk["text"],
            }
        )

    return {
        "ok": True,
        "query": query,
        "results": results
    }


# =========================================================
# TOOL DEFINITION
# =========================================================

SEARCH_CORPUS_TOOL = {
    "type": "function",

    "function": {
        "name": "search_corpus",

        "description": (
            "Search the EU AI Act and NIST AI Risk Management "
            "Framework documents. Use focused terminology likely "
            "to appear in the documents. If a search does not "
            "provide enough evidence, search again using substantially "
            "different terminology rather than minor rewording."
        ),

        "parameters": {
            "type": "object",

            "properties": {

                "query": {
                    "type": "string",
                    "description": (
                        "Focused document search query using important "
                        "terms, headings, concepts, abbreviations, or "
                        "likely source terminology."
                    ),
                },

                "k": {
                    "type": "integer",
                    "description": (
                        f"Number of passages. "
                        f"Minimum 1, maximum {MAX_K}. "
                        f"Default {DEFAULT_K}."
                    ),
                },
            },

            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


TOOLS = [SEARCH_CORPUS_TOOL]

TOOL_FUNCTIONS = {
    "search_corpus": search_corpus
}


# =========================================================
# RUN TOOL
# =========================================================

def run_tool(name: str, arguments: str | dict) -> dict:

    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        return {
            "ok": False,
            "error": f"Unknown tool: {name}"
        }

    if isinstance(arguments, str):

        try:
            arguments = json.loads(arguments or "{}")

        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "Tool arguments are not valid JSON."
            }

    if not isinstance(arguments, dict):
        return {
            "ok": False,
            "error": "Tool arguments must be a JSON object."
        }

    try:
        return function(**arguments)

    except TypeError as error:
        return {
            "ok": False,
            "error": f"Invalid arguments: {error}"
        }


# =========================================================
# TERMINAL DEBUGGING
# =========================================================

def print_search_results(result: dict, attempt: int):

    print(
        f"\n========== SEARCH ATTEMPT {attempt} =========="
    )

    if not result.get("ok"):

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        print(
            "=============================================="
        )

        return

    print("Query:", result.get("query"))

    for item in result.get("results", []):

        print(
            f"\n--- CHUNK {item['ref']} ---"
        )

        print("Source:", item["source"])
        print("Page:", item["page"])
        print("Score:", item["score"])
        print("Text:", item["text"])

    print(
        "\n=============================================="
    )


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an AI Governance RAG assistant.

Your document corpus contains:
1. The EU AI Act
2. NIST AI Risk Management Framework 1.0

DOCUMENT QUESTIONS:

If the user's question asks about information contained in either
document, you MUST use search_corpus before answering.

The retrieved passages are the ONLY evidence you may use in the
final answer.

Do not use your background knowledge as evidence for document claims.


RETRIEVAL STRATEGY:

The user's wording may differ from the terminology in the documents.

Create focused search queries rather than automatically copying the
entire user question.

After every search, inspect the retrieved passages.

Ask yourself only whether those passages contain sufficient evidence
to answer the user's question.

If they do, stop searching and answer.

If they do not, search again.


IMPORTANT QUERY REFORMULATION RULE:

A retry MUST be substantially different from the previous search.

Do NOT make weak reformulations such as:

"core functions NIST AI RMF"
"functions of AI RMF"
"AI RMF core functions list"

Those are essentially the same search.

Instead, change to terminology that is likely to occur literally in
the source document.

You may use general knowledge ONLY for creating candidate search
keywords.

Candidate search keywords are NOT evidence.

For example, when searching for the four NIST AI RMF functions, a
useful substantially different query is:

"GOVERN MAP MEASURE MANAGE"

You must then verify those terms against retrieved passages before
using them in the answer.

Other retries should similarly use named concepts, headings,
abbreviations, article numbers, section terminology, or source-specific
vocabulary.

Maximum searches: 3.


CITATION RULES:

Use only information supported by retrieved passages.

Preserve official terminology from the retrieved source.

For example, write:

GOVERN
MAP
MEASURE
MANAGE

when those are the terms appearing in the source.

Only cite source filenames and page numbers that appear in the
metadata of retrieved passages that actually support the claim.

Never invent a page number.

Use this citation format:

[nist_ai_rmf_1_0.pdf, p. 26]

or:

[eu_ai_act_2024.pdf, p. 123]


OUT-OF-SCOPE QUESTIONS:

If a question is unrelated to the EU AI Act or NIST AI RMF,
do NOT search the corpus.

Respond exactly:

I don't have enough information in the provided documents to answer this question.


INSUFFICIENT EVIDENCE:

If three searches still do not provide enough document evidence,
respond exactly:

I don't have enough information in the provided documents to answer this question.
"""


# =========================================================
# AGENT
# =========================================================

def run_agent(question: str):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": question
        }
    ]


    # This trace contains ONLY observable agent actions.
    trace = []

    search_attempts = 0
    did_search = False


    while True:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        message = response.choices[0].message


        print(
            "\n========== MODEL DECISION =========="
        )

        print(message)

        print(
            "===================================="
        )


        # =================================================
        # TOOL CALL
        # =================================================

        if message.tool_calls:

            if search_attempts >= MAX_SEARCH_ATTEMPTS:
                break


            messages.append(message)


            for tool_call in message.tool_calls:

                if search_attempts >= MAX_SEARCH_ATTEMPTS:

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "ok": False,
                                    "error": (
                                        "Maximum search attempts reached."
                                    )
                                }
                            )
                        }
                    )

                    continue


                did_search = True
                search_attempts += 1


                tool_name = tool_call.function.name
                tool_arguments = tool_call.function.arguments


                # -----------------------------------------
                # PARSE QUERY
                # -----------------------------------------

                try:

                    parsed = json.loads(
                        tool_arguments or "{}"
                    )

                    search_query = parsed.get(
                        "query",
                        ""
                    )

                    search_k = parsed.get(
                        "k",
                        DEFAULT_K
                    )

                except Exception:

                    search_query = str(
                        tool_arguments
                    )

                    search_k = DEFAULT_K


                # -----------------------------------------
                # SAVE SEARCH ACTION
                # -----------------------------------------

                trace.append(
                    {
                        "type": "search",
                        "attempt": search_attempts,
                        "tool": tool_name,
                        "query": search_query,
                        "k": search_k,
                    }
                )


                # -----------------------------------------
                # RUN TOOL
                # -----------------------------------------

                result = run_tool(
                    tool_name,
                    tool_arguments
                )


                print_search_results(
                    result,
                    search_attempts
                )


                # -----------------------------------------
                # SAVE RETRIEVAL RESULT
                # -----------------------------------------

                if result.get("ok"):

                    matches = []

                    for item in result.get(
                        "results",
                        []
                    ):

                        matches.append(
                            {
                                "source": item["source"],
                                "page": item["page"],
                                "score": item["score"],
                            }
                        )


                    trace.append(
                        {
                            "type": "retrieval",
                            "attempt": search_attempts,
                            "count": len(matches),
                            "matches": matches,
                        }
                    )

                else:

                    trace.append(
                        {
                            "type": "retrieval_failed",
                            "attempt": search_attempts,
                            "message": result.get(
                                "error",
                                "No useful passages found."
                            ),
                        }
                    )


                # -----------------------------------------
                # RETURN RESULT TO MODEL
                # -----------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False
                        ),
                    }
                )


            continue


        # =================================================
        # MODEL ANSWERED WITHOUT ANOTHER TOOL
        # =================================================

        if did_search:

            trace.append(
                {
                    "type": "final",
                    "searched": True,
                    "search_count": search_attempts,
                    "message": (
                        "Search completed. Final answer generated "
                        "from the retrieved document evidence."
                    )
                }
            )

        else:

            trace.append(
                {
                    "type": "no_search",
                    "searched": False,
                    "message": (
                        "No document search was performed because "
                        "the question was outside the document scope."
                    )
                }
            )


        print(
            "\n========== FINAL ANSWER =========="
        )

        print(message.content)

        print(
            "=================================="
        )


        return message, trace


    # =====================================================
    # MAXIMUM SEARCHES REACHED
    # =====================================================

    messages.append(
        {
            "role": "system",
            "content": (
                "The maximum number of searches has been reached. "
                "Do not call another tool. "
                "Answer using ONLY the retrieved passages already "
                "provided in this conversation. "
                "Use exact source terminology. "
                "Only cite source and page metadata from passages "
                "that support the claim. "
                "If the passages are insufficient, respond exactly: "
                "\"I don't have enough information in the provided "
                "documents to answer this question.\""
            )
        }
    )


    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )


    final_message = (
        final_response
        .choices[0]
        .message
    )


    trace.append(
        {
            "type": "final",
            "searched": True,
            "search_count": search_attempts,
            "message": (
                "Maximum search attempts reached. "
                "The final response was generated using only "
                "the evidence already retrieved."
            )
        }
    )


    print(
        "\n========== FINAL ANSWER =========="
    )

    print(final_message.content)

    print(
        "=================================="
    )


    return final_message, trace


# =========================================================
# DIRECT RETRIEVAL TEST
# =========================================================

def main():

    query = (
        " ".join(sys.argv[1:])
        or
        "GOVERN MAP MEASURE MANAGE"
    )


    result = search_corpus(
        query
    )


    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()