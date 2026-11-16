# Data card — SelfCheck Agent

This document describes **what data the project uses, why, how it is collected and labelled,
and what its known limitations are.** Numbers below come from `reports/data_stats.md`.

## 1. Why this data

The project's claims are measurements: *how often the verifier catches a wrong answer
(true-positive flags), how often it flags a correct answer (false positives), and how
accuracy changes with vs. without verification.* All three need, for every question,

1. a **gold answer** — to decide whether the agent was right,
2. **gold evidence passages** — to tell retrieval failures apart from reasoning failures, and to check whether the verifier's evidence check agrees with the truth,
3. **the correct behaviour** — answer, **abstain** (unanswerable) or **ask for clarification** (ambiguous).

So we use established QA benchmarks with human-written gold labels, not scraped documents.

## 2. Sources (Tier 1: benchmark data)

| Dataset | Split used | What it gives us | Behaviour tested | License |
|---|---|---|---|---|
| **MuSiQue** (full v1.0) | dev | 2–4 hop questions, **gold decomposition with a gold answer + supporting paragraph per hop**, 20 passages each (gold + distractors); every answerable question has an unanswerable twin whose key paragraph was removed | per-step verification, abstention | CC BY 4.0 |
| **HotpotQA** (distractor) | validation | 2-hop bridge/comparison questions, 2 gold + 8 distractor paragraphs, sentence-level supporting facts | separating real evidence from look-alike distractors | CC BY-SA 4.0 |
| **Qasper** v0.3 (official AllenAI release) | dev papers + test papers | questions over **full NLP research papers**, written from title+abstract only (so ~9% are unanswerable from the paper), multiple annotators, evidence paragraphs | long technical documents, abstention | CC BY 4.0 |
| **AmbigQA** (full) | validation | questions labelled single-answer vs. **ambiguous**, with disambiguated questions + answers; passages = search snippets annotators saw | ask-for-clarification | CC BY-SA 3.0 |

All downloads are pinned to an exact revision and verified by SHA-256 (`src/selfcheck/data/sources.py`).
Only official evaluation splits are used; no model is trained on them.

## 3. Unified schema (`src/selfcheck/data/schema.py`)

- **Document**: `doc_id` (content hash, de-duplicated per dataset), `title`, `text`.
- **Example**: `question`, `answers` (all acceptable aliases / annotator answers), `answer_type`
  (`span | yesno | free | none`), `answerable`, `ambiguous`, `gold_doc_ids`, `scope_doc_ids`
  (the question's own passage pool), `steps` (gold hops: sub-question, answer, gold passage),
  `disambiguations`, `meta`.

## 4. Sampling

Deterministic, stratified (seed 13, `configs/data.yaml`):

| Dataset | dev (tuning only) | test (held out, reported) | stratified by |
|---|---|---|---|
| MuSiQue | 200 | 500 | answerable × hops |
| HotpotQA | 200 | 500 | type × level |
| Qasper | 200 | 500 | answer type — **paper-level split** (no paper in both) |
| AmbigQA | 100 | 200 | ambiguous |

`data/splits/*.json` (committed) lists the exact question ids. Sizes are moderate because each
question costs several LLM calls (generator, critic, self-consistency samples) on a free API tier.

## 5. Labelling decisions

- **Qasper answerability** = majority of annotators (ties → first annotator). Gold answers = union of answering annotators. Evidence that is a figure/table (`FLOAT SELECTED`) cannot be retrieved as text and is counted in `meta.float_evidence`; 3–5% of answerable questions therefore have no text evidence.
- **AmbigQA ambiguity** = majority over annotators' `singleAnswer` / `multipleQAs` labels. Ambiguous questions keep the original NQ answer in `answers` and the interpretations in `disambiguations`.
- **MuSiQue unanswerable** questions have `answers = []`; the original answer is kept in `meta.original_answer` for analysis only.

## 6. Known limitations

- **HotpotQA validation contains only "hard" questions**, so the level stratum is constant.
- **MuSiQue sub-questions** are sometimes knowledge-base style (`"MYSTIC >> creator"`). The agent writes its own decomposition; gold hops are used only to *grade* steps.
- **AmbigQA has no passage-level evidence labels** and its passages are short search snippets. It is used only to evaluate the clarification decision, not evidence verification.
- MuSiQue questions are built from shared single-hop questions, so dev and test can share some passages. This is acceptable because nothing is trained on dev beyond a few thresholds and a small confidence combiner.
- These are English, Wikipedia- or NLP-paper-domain datasets; results may not transfer to other domains.

## 7. Data produced later in the pipeline (Tiers 2–4)

| Tier | Produced by | Content |
|---|---|---|
| 2. Agent run logs | running the agent **with verification off** on dev+test | per step: sub-question, retrieved passages, draft answer, reasoning, tokens, latency |
| 2b. Correctness labels | exact match / token F1 + LLM judge (`gpt-oss-120b`) | correct / incorrect per step and final answer |
| 3. Human audit | ~200 hand-labelled outputs | judge-vs-human agreement (Cohen's κ) |
| 4. Controlled perturbations | corrupting correct runs | gold passage removed, entity/number swapped, contradicting distractor added — label known by construction |
