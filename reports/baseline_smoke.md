# Baseline agent — smoke test (dev, verification OFF)

Small sanity run of the baseline agent (`gpt-oss-20b`, hybrid+rerank retrieval, top-5) to validate the
pipeline end to end and measure API cost per question. **Not the reported results** — those come from
the full dev/test runs with LLM-judge labels.

| dataset | n | EM | token F1 | abstained | LLM calls / q | tokens / q | errors |
|---|---|---|---|---|---|---|---|
| ambigqa | 8 | 0.88 | 0.88 | 12% | 2.4 | 1387 | 0 |
| hotpotqa | 10 | 0.70 | 0.92 | 0% | 2.8 | 2403 | 0 |
| musique | 16 | 0.69 | 0.76 | 62% | 3.7 | 3522 | 0 |
| qasper | 8 | 0.25 | 0.41 | 25% | 2.4 | 1911 | 0 |

Observations:
- Many F1 < 1 cases are paraphrases ("Ghanaian" vs "a Ghanaian boxer") → an LLM judge is needed for correctness labels.
- Qasper is hardest: retrieval recall@5 is 0.59 and answers are long/abstractive.
- False abstentions occur (MuSiQue "state" vs "district") — candidate cases for verifier-triggered retry.
- First prompt version over-abstained (5/6 on MuSiQue) because the planner guessed entities from memory and the
  finaliser refused near-synonyms; both fixed in `agent/prompts.py` (tuned on dev only).
