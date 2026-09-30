# One Piece — RAG Project

An AI regulation assistant that answers questions about the
**EU Artificial Intelligence Act (Regulation (EU) 2024/1689)** and the
**NIST AI Risk Management Framework 1.0**. It is built from scratch
(no LangChain or LlamaIndex): it searches the two documents for the most
relevant passages, then asks `gpt-4o-mini` to answer **only** from those
passages, with a citation to the document and page for every claim.
Questions outside the documents get a clear "I don't have enough
information" reply instead of a made-up answer.

## Team Members

1. Aravind Reddy Kolan
2. Prem Siva Naga Sai Ram Kumar Gudivada
3. Praveen Kumar Koppula

## How it works

The project has two parts, both served by the same web app.

### 1. RAG pipeline (fixed steps)

| Step | Script | Output |
|------|--------|--------|
| 1. Extract text from the PDFs, one entry per page | `scripts/extract.py` | `data/corpus.json`, `data/sample.json` |
| 2. Split pages into 500-character chunks (100 overlap) | `app/chunking.py` | `data/chunks.json` |
| 3. Embed the chunks with `text-embedding-3-small` and build a FAISS index | `app/build_index.py` | `data/my_index.faiss` |
| 4. Retrieve the 5 best chunks with **hybrid search** (vectors + BM25 keywords, merged with Reciprocal Rank Fusion) | `app/retrieval.py` | — |
| 5. Generate a cited answer with `gpt-4o-mini` (temperature 0) | `app/generate_answer.py` | — |

Hybrid search keeps exact terms (Article numbers, GOVERN / MAP / MEASURE /
MANAGE) findable and stops short title chunks from crowding out the real
answer. Set `SEARCH_MODE = "vector"` in `app/retrieval.py` to compare.

### 2. Agent (the model chooses its tools)

`app/agent.py` runs an agent loop from scratch (no frameworks):
think → call tools → read results → repeat, with a hard cap of 6 model
calls, temperature 0, and every call's cost tracked.

| Tool | File | What it does |
|------|------|--------------|
| `search_corpus` | `app/agent.py` | Wraps `retrieve()`: passages with source, page and score. Says "rephrase" when nothing relevant is found. |
| `calculator` | `app/calculator.py` | Exact arithmetic with `simpleeval`. |
| `quote_exact` | `app/quote_exact.py` | Full stored text of one page of a named document. |
| `get_article` | `app/get_article.py` | Full text of an EU AI Act Article (or one paragraph, e.g. 99(3)). |
| `date_tool` | `app/date_tool.py` | Today's date and days between two dates. |
| `count_term` | `app/term_counter.py` | Exact count of a word or phrase per document and page. |
| `send_message` | `app/send_message.py` | **Proposes** a Slack/Teams message. Nothing is sent until a person answers **Yes**. |

- Several independent tools can run in the same step (shown as
  "⚡ 2 tools in parallel" in the app).
- If a search finds nothing useful, the agent rephrases and searches
  again; if the documents still don't contain the answer it says so.
- Identical repeated tool calls reuse the first result.
- Every run (tool, arguments, result) is saved to `data/trajectories.jsonl`.

`app/multi_agent.py` adds an optional **multi-agent team**: a coordinator
that asks an EU AI Act specialist agent and a NIST AI RMF specialist agent
(each only sees its own document) and combines their cited answers.

Every API call's estimated cost is recorded by `app/cost_tracker.py`
in `data/cost_tracker.json`.

## Project structure

```
one-piece-rag/
├── README.md
├── requirements.txt
├── .env                      ← API key + Slack webhook (never committed)
├── app/
│   ├── app.py                ← Gradio web app (agent, multi-agent, RAG modes)
│   ├── agent.py              ← agent loop, tools registry, trajectory log
│   ├── multi_agent.py        ← coordinator + EU / NIST specialist agents
│   ├── chunking.py
│   ├── build_index.py
│   ├── retrieval.py          ← hybrid search (FAISS + BM25)
│   ├── generate_answer.py
│   ├── cost_tracker.py
│   ├── calculator.py         ← tool: safe arithmetic (simpleeval)
│   ├── quote_exact.py        ← tool: exact text of a document page
│   ├── get_article.py        ← tool: EU AI Act Article by number
│   ├── date_tool.py          ← tool: dates
│   ├── term_counter.py       ← tool: count_term
│   └── send_message.py       ← tool: Slack/Teams message with approval
├── scripts/
│   ├── extract.py
│   ├── EXTRACTION_REPORT.md
│   ├── retrieval_checker.py  ← inspect retrieval for one question
│   ├── eval_runner.py        ← run all questions (RAG or agent)
│   └── FAILURE_LOG.md
└── data/                     ← gitignored, shared via the Teams channel
    ├── sources/              ← eu_ai_act_2024.pdf, nist_ai_rmf_1_0.pdf
    ├── corpus.json, chunks.json, my_index.faiss, embeddings.npy
    ├── questions.json        ← 40 evaluation questions
    ├── multistep_questions.json ← 6 questions that force tool use
    └── trajectories.jsonl    ← every agent run
```

## Installation

Requires **Python 3.12** and **uv**.

```bash
git clone https://github.com/premgudivada7/one-piece-rag.git
cd one-piece-rag
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...   # for send_message
```

`send_message` can also use Slack's official MCP server
(`SLACK_MCP_TOKEN` + `SLACK_CHANNEL_ID`) or a Teams webhook
(`TEAMS_WEBHOOK_URL`); see the top of `app/send_message.py`.
Check the connection with `python app/send_message.py --check`.

Download the `data/` folder from the team's Teams channel and place it
in the project root. It already contains the chunks and the FAISS index,
so you do **not** need to embed anything.

## Running the app

```bash
python app/app.py
```

Open http://127.0.0.1:7860, choose a mode, type a question and press **Enter**.

- **🤖 Single agent**: the answer, a "Tools used" line, and the
  **Agent Steps** panel with every step and a "View code" dropdown per tool call.
- **👥 Multi-agent team**: the coordinator's steps plus each specialist's steps.
- **📚 RAG pipeline (no agent)**: the original retrieve → generate pipeline
  with the retrieved passages.

Ask the agent to share an answer ("... and send the answer to our Slack
channel"): it proposes the message, the app shows it with
**✅ Yes, send it / ❌ No, cancel**, and only Yes posts it.

From the terminal:

```bash
python app/agent.py "What is the fine in Article 99(3), and what is half of it?"
python app/multi_agent.py "Do the EU AI Act and the NIST AI RMF have the same legal status?"
```

## Rebuilding the data (only if the sources change)

```bash
python scripts/extract.py      # PDFs → corpus.json
python app/chunking.py         # corpus.json → chunks.json
python app/build_index.py      # chunks.json → my_index.faiss (paid API calls)
```

`build_index.py` refuses to reuse an old index if `chunks.json` has
changed; move the old `index_metadata.json`, `embeddings.npy` and
`my_index.faiss` out of `data/` first.

## Evaluation

```bash
# One question: shows retrieved chunks and ✅/❌ for the expected pages
python scripts/retrieval_checker.py "What is a high-risk AI system?"
python scripts/retrieval_checker.py --id 3

# RAG pipeline on all 40 questions → data/eval_results.json
python scripts/eval_runner.py --retrieval-only                  # hit rate only (cheapest)
python scripts/eval_runner.py --retrieval-only --search vector  # compare with vector-only
python scripts/eval_runner.py --judge                           # + answers scored by gpt-4o-mini

# Agent → data/eval_results_agent.json
python scripts/eval_runner.py --agent --judge
python scripts/eval_runner.py --agent --judge --questions data/multistep_questions.json
```

For the agent, a "retrieval hit" means an expected page was returned by
one of its tools or cited; multi-step questions also check that the
expected tools (e.g. `calculator`, `date_tool`) were used.
Failures and fixes are recorded in `scripts/FAILURE_LOG.md`.

## Checking the budget

```bash
python app/cost_tracker.py
```

## Known limitations

- Tables are extracted as plain text, so their row order can be lost.
- Paragraphs that continue onto the next PDF page are split across chunks.
- Page numbers are PDF page positions, which can differ from the numbers
  printed on the pages.
- The chat does not remember earlier questions; each question is answered on its own.
- `send_message` uses the Slack webhook fallback; the Slack MCP path is
  implemented but needs an internal Slack app with OAuth (`chat:write`).
