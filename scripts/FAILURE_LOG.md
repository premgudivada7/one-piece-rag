# Failure Log

Debugging history for the One Piece RAG assistant. Every failed
evaluation question is recorded here, with its cause, the fix we tried
and whether the fix worked.

## How we use this log

1. Run `python scripts/eval_runner.py` to see which questions fail.
2. For each failure, run `python scripts/retrieval_checker.py --id N`:
   - expected pages missing (❌) → **retrieval problem** (chunking, top-k, search)
   - expected pages found (✅) but a wrong answer → **generation problem** (prompt)
3. Apply one fix, then re-run the eval runner on **all** questions to check
   the fix worked and nothing else broke.
4. Add a row below. If a fix breaks another question, record that too.

Settings at the start: chunk size 500, overlap 100,
`text-embedding-3-small`, FAISS `IndexFlatIP`, top-k 5, `gpt-4o-mini`
at temperature 0.

## Failures

| # | Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|----------|----------|:---:|---------|-------------|:---:|
| 1 | (Agent) "What is the maximum fine under Article 99 for prohibited AI practices, and what is half of it?" | multi-step | ✅ (p.115 retrieved at rank 3) | Agent answered EUR 1 500 000 (Article 100, fines on EU institutions, p.117) instead of EUR 35 000 000 / 7 % (Article 99(3), p.115). The top 500-char chunk had no Article heading, so the model trusted the wrong passage. | Added system-prompt rule 7: when the question names an Article or passages give different values, verify with quote_exact before answering. Re-run: agent quoted p.115 and p.117, answered EUR 35 000 000 → half EUR 17 500 000 (p.115). | ✅ |
| 2 | (Agent) "What is the fine in Article 99(3), what is half of it, and how many days ago did the EU AI Act start to apply generally?" | multi-step | ✅ (get_article 99(3), p.115) | Fine and half were correct, but the agent called date_tool in the same step as get_article, with a start date from memory (1 May 2024, uncited) instead of Article 113 (2 August 2026, p.123): answered 881 days instead of 58. It also sent a placeholder 'fine_value / 2' to the calculator. | System prompt: only parallelise independent tools, never placeholders/guessed values; rule 8: dates for date_tool must come from a tool result (Article 113 via get_article) and be cited. Same rule added to date_tool description and coordinator prompt. Re-run: get_article(113) read first, date_tool 2026-08-02 → today = 58 days (p.123), no placeholders. | ✅ |

## Eval runs

| Date | Change | Retrieval hit | Pass | Fail | Notes |
|------|--------|:---:|:---:|:---:|-------|
| | Baseline | | | | |
