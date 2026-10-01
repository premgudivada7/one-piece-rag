# Agent failure log

How we work: run the agent eval with logging on, read each failed trace in `data/traces/` (or with
`python scripts/view_trace.py`), write down the **first** thing that went wrong (an early wrong turn
causes most of what follows), group the notes into categories, count them, and fix the biggest
category first. Every category becomes an eval question we keep.

## Failed runs, first wrong turn

| # | Run | Question | First thing that went wrong | Category |
|---|-----|----------|-----------------------------|----------|
| 1 | 29 Sep, first agent | "What are the four core functions of the NIST AI RMF?" | Five near-identical searches that all returned the same title pages, then the step cap and "I don't have enough information" | Kept searching instead of answering |
| 2 | 30 Sep, first agent | "Maximum fine in Article 99 for prohibited practices, and half of it?" | Trusted a passage from Article 100 (fines on EU institutions) instead of Article 99(3) | Trusted the wrong passage |
| 3 | 30 Sep, first agent | "Fine in Article 99(3), half of it, and how many days since the Act applies?" | Called date_tool with a start date from memory (1 May 2024) before reading Article 113 | Used a value it had not read |
| 4 | 30 Sep, first agent | "What are the four core functions…?" (re-run) | Added a description of each function that is not on the cited page | Unsupported text under a real citation |
| 5 | 30 Sep, first agent | "Maximum fine in Article 99(3), and half of it?" | Sent a guessed number to the calculator in the same step as get_article | Used a value it had not read |
| 6 | 30 Sep, first agent | "How often does each document use 'trustworthy', and the total?" | Added 17 + 23 in its head | Did the math in its head |
| 7 | 30 Sep, first agent | "If an AI is open about how it works, is it accurate and fair?" | Rewrote the question into keywords that missed page 21 | Rewrote the search away from the user's words |
| 8 | 30 Sep, first agent | "From which date does the Act apply, and how many days ago?" | Read date_tool's "+59" as "59 days away" | Misread a tool result |
| 9 | 1 Oct, Grok, multi-step q02 | Fine figures, then a difference | Right numbers, computed in its head | Did the math in its head |
| 10 | 1 Oct, Grok, multi-step q03 | Fine figures, then half | Same | Did the math in its head |
| 11 | 1 Oct, Grok, q38 "What are the requirements?" | Ambiguous question | Searched, then read Articles 8, 9 and 10 one by one: 6 model calls, step cap | Kept searching instead of answering |
| 12 | 1 Oct, Grok, q40 "Who is responsible?" | Ambiguous question | Five searches and three get_article calls, step cap | Kept searching instead of answering |
| 13 | 1 Oct 22:29, after reranking, q21 legal status | Priority question | Its first answer quoted "Regulation (EU) 2024/1689" from memory, a number no tool had returned. The number check sent it back, it quoted 6 pages to verify and hit the step cap (6 calls, 10 s). Final answer correct and cited | Used a value it had not read |
| 14 | 1 Oct 22:35, q46 "What can you help me with?" (no tool) | No-tool scenario | It answered right away, but wrote "EU AI Act 2024"; the number check didn't know "2024" comes from the file name, sent it back, and it searched once for nothing | A guard of ours misfired |

Not agent failures (we checked the question and the grading first, as the course says):
q47 ("Send this answer to our team channel") on 1 Oct 22:35 was marked partial because the judge only saw
the final "the message was not sent", not the message the agent proposed (which was right and cited): the
judge now sees the proposed message too. q06 on 1 Oct was marked "partial" only because the answer added correct details from Article 12(2)
(judge prompt fixed); two multi-step questions expected `search_corpus` where `get_article` was just as
good (`expected_tools` now accepts `"search_corpus|get_article"`); date answers written as "59 days" go
stale the next day (expected answers now use `{days_since:2026-08-02}`); a correct refusal was graded as
a "false refusal" because the question was not labelled as one to refuse (the runner now also treats an
expected answer that is itself a refusal as "should refuse").

## Categories and counts

| Category | Count | Where the fix lives | Fix | Eval question we keep |
|----------|:-----:|---------------------|-----|-----------------------|
| Kept searching instead of answering | 3 | prompt + harness | Repeated-call block, "rephrase at most twice", and (1 Oct) a rule for vague questions: search once, say what is unclear, ask | q38, q40 (`--ids q38,q40`) and the out-of-scope agent question |
| Did the math in its head | 3 | code | Number check: a number in the final answer that no tool returned triggers one reminder to use the calculator | the search-then-calculate agent question |
| Used a value it had not read | 3 | prompt + tool descriptions + number check | Only dates and numbers read in a tool result; the number check catches the rest (it did on q21, at the cost of extra steps) | the date_tool agent question, q21 |
| Trusted the wrong passage | 1 | retrieval | get_article reads the exact Article; reranking puts the right passage first | the get_article agent question |
| Rewrote the search away from the user's words | 1 | retrieval | Hybrid search + reranking on the model's query | q32 through the agent |
| Unsupported text under a real citation | 1 | prompt | "Every sentence must come from a tool result" | q10 through the agent |
| Misread a tool result | 1 | tool | date_tool answers with a sentence ("was 59 days ago, in the past") | the date_tool agent question |
| A guard of ours misfired | 1 | code | The number check now also accepts numbers from the document names | q46 |

The top two categories were tied. "Did the math in its head" was fixed first because it gives wrong
numbers silently; "Kept searching" is the open one, fixed on 1 Oct with the vague-question rule.

## Regression checks

After every change: priority questions first, then the agent questions.

```
python scripts/eval_runner.py --mode agent --priority --judge
python scripts/eval_runner.py --mode agent --judge --compare data/eval_runs/<previous>_agent.json
```

| Date | Change | Priority (5) | Agent questions | Regressions |
|------|--------|:---:|:---:|---|
| 1 Oct | Number check | 5/5 | 39/40 (old 40-question run) | none |
| 1 Oct 22:29 | Reranking, 3 passages per search, vague-question rule, injection hardening split out | 5/5 | not written yet | none (q21 now takes 6 calls, see #13) |
| 1 Oct 22:35 | 7 agent questions added (q41–q47) | 5/5 | 6/7 (q47, grading) | none |
| 1 Oct 22:40 | Number check accepts document names; judge sees the proposed message | 5/5 | 7/7 (q47 fixed, q46 now uses no tool) | none |
