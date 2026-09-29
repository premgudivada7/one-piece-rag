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

| Step | Script | Output |
|------|--------|--------|
| 1. Extract text from the PDFs, one entry per page | `scripts/extract.py` | `data/corpus.json`, `data/sample.json` |
| 2. Split pages into 500-character chunks (100 overlap) | `app/chunking.py` | `data/chunks.json` |
| 3. Embed the chunks with `text-embedding-3-small` and build a FAISS index | `app/build_index.py` | `data/my_index.faiss` |
| 4. Embed the question and retrieve the 5 closest chunks | `app/retrieval.py` | — |
| 5. Generate a cited answer with `gpt-4o-mini` (temperature 0) | `app/generate_answer.py` | — |
| 6. Chat web interface | `app/app.py` | http://127.0.0.1:7860 |

Every API call's estimated cost is recorded by `app/cost_tracker.py`
in `data/cost_tracker.json`.

## Project structure

```
one-piece-rag/
├── README.md
├── requirements.txt
├── .env                      ← your OpenAI key (never committed)
├── app/
│   ├── app.py                ← Gradio chat interface
│   ├── chunking.py
│   ├── build_index.py
│   ├── retrieval.py
│   ├── generate_answer.py
│   ├── cost_tracker.py
│   ├── calculator.py         ← agent tool: safe arithmetic (simpleeval)
│   └── quote_exact.py        ← agent tool: exact text of a document page
├── scripts/
│   ├── extract.py
│   ├── EXTRACTION_REPORT.md
│   ├── retrieval_checker.py  ← inspect retrieval for one question
│   ├── eval_runner.py        ← run all evaluation questions
│   └── FAILURE_LOG.md
└── data/                     ← gitignored, shared via the Teams channel
    ├── sources/              ← eu_ai_act_2024.pdf, nist_ai_rmf_1_0.pdf
    ├── corpus.json, chunks.json, my_index.faiss, embeddings.npy
    └── questions.json        ← 40 evaluation questions
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
```

Download the `data/` folder from the team's Teams channel and place it
in the project root. It already contains the chunks and the FAISS index,
so you do **not** need to embed anything.

## Running the app

```bash
python app/app.py
```

Open http://127.0.0.1:7860, type a question and press **Enter**.
A "Sources" panel shows the passages cited in the answer and all
retrieved passages, with document, page and similarity score.

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

# All 40 questions from data/questions.json → data/eval_results.json
python scripts/eval_runner.py --retrieval-only   # retrieval hit rate only (cheapest)
python scripts/eval_runner.py                    # + generated answers
python scripts/eval_runner.py --judge            # + gpt-4o-mini as judge
python scripts/eval_runner.py --priority         # only the 5 priority questions
```

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
