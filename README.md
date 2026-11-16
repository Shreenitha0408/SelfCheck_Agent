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
| 2. Retrieval (chunking, embeddings, reranking) | ⏳ |
| 3. Baseline agent (no verification) + run logs | ⏳ |
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

Small models (embeddings, reranker, NLI) run locally on CPU.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python scripts/prepare_data.py   # download (pinned + checksummed), normalise, split, report
pytest -q
```
