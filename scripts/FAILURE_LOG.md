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
| | | | | | | |

## Eval runs

| Date | Change | Retrieval hit | Pass | Fail | Notes |
|------|--------|:---:|:---:|:---:|-------|
| | Baseline | | | | |
