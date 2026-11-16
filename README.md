# SelfCheck Agent

A multi-step document-QA agent that **verifies each step before moving on**. Every intermediate
answer is checked against the retrieved evidence and cross-checked by an independent critic
model, then given a calibrated confidence score. When confidence is low, the agent
**retries, flags the answer as uncertain, abstains, or asks for clarification** instead of
presenting a shaky answer as fact.

The point of the project is the **measurement**, not just the demo:

- verifier **true-positive rate** (wrong answers caught) and **false-positive rate** (correct answers wrongly flagged)
- **accuracy with vs. without** the verification layer, on held-out data
- calibration (ECE), AUROC of the confidence score, risk–coverage curves, and extra latency/token cost

## Status

| Stage | State |
|---|---|
| 1. Data collection & preparation | ✅ done — see [DATA.md](DATA.md), [reports/data_stats.md](reports/data_stats.md) |
| 2. Retrieval (chunking, embeddings, reranking) | ✅ done — [reports/retrieval.md](reports/retrieval.md) |
| 3. Baseline agent (no verification) + run logs | 🟡 built + smoke-tested — [reports/baseline_smoke.md](reports/baseline_smoke.md); full runs pending |
| 4. Correctness labelling (auto + LLM judge + human audit) | ⏳ |
| 5. Verifier: NLI evidence check, LLM critic, self-consistency → calibrated confidence | ⏳ |
| 6. Decision policy (accept / retry / flag / abstain / clarify) | ⏳ |
| 7. Evaluation, ablations, perturbation stress tests | ⏳ |
| 8. Deployment (FastAPI + Gradio on Hugging Face Spaces, Docker, CI) | ⏳ |

## Data

| Dataset | Tests | dev / test |
|---|---|---|
| MuSiQue | multi-hop reasoning with per-hop gold answers; unanswerable twins | 200 / 500 |
| HotpotQA (distractor) | real vs. look-alike evidence | 200 / 500 |
| Qasper | long research papers; abstention | 200 / 500 |
| AmbigQA | when to ask for clarification | 100 / 200 |

## Models (all open-weight, served via Groq's free tier)

| Role | Model |
|---|---|
| Generator (the agent) | `openai/gpt-oss-20b` |
| Critic (independent check, different model family) | `qwen/qwen3.8-27b` |
| Judge (grading correctness for evaluation) | `openai/gpt-oss-120b` |

Small models run locally on CPU via ONNX (`fastembed`): `BAAI/bge-small-en-v1.5` embeddings and the
`ms-marco-MiniLM-L-6-v2` cross-encoder reranker. Retrieval = BM25 + dense fused with reciprocal-rank fusion,
then reranked; it finds the gold passage for 98.9% of MuSiQue hops in the top 5.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python scripts/prepare_data.py   # download (pinned + checksummed), normalise, split, report
pytest -q
python scripts/eval_retrieval.py                       # retrieval recall on dev (~45 min first time: embeds corpora)
python scripts/run_agent.py --dataset hotpotqa --split dev --limit 10 --run smoke
```

All LLM calls are cached in `.cache/llm.sqlite`; runs are resumable and stop cleanly at the daily API quota.
