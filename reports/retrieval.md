# Retrieval evaluation (dev split, answerable questions)

Scoped retrieval over each question's own passage pool. Chunks: ≤180 words, 40-word overlap.
`recall@k` = share of gold passages in top-k; `all_gold@k` = all gold passages found;
`step_recall@k` = per-hop gold passage found when querying with the gold sub-question (MuSiQue).

## musique (n=100)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 | step_recall@1 | step_recall@3 | step_recall@5 | step_recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.244 | 0.447 | 0.527 | 0.683 | 0.000 | 0.080 | 0.140 | 0.360 | 0.679 | 0.887 | 0.940 | 0.977 |
| dense | 0.312 | 0.595 | 0.667 | 0.841 | 0.000 | 0.230 | 0.340 | 0.610 | 0.823 | 0.958 | 0.970 | 0.996 |
| hybrid | 0.305 | 0.528 | 0.606 | 0.761 | 0.000 | 0.140 | 0.240 | 0.450 | 0.796 | 0.943 | 0.966 | 1.000 |
| hybrid_rerank | 0.342 | 0.545 | 0.630 | 0.763 | 0.000 | 0.170 | 0.260 | 0.490 | 0.868 | 0.970 | 0.989 | 1.000 |

## hotpotqa (n=200)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.445 | 0.728 | 0.828 | 0.985 | 0.000 | 0.495 | 0.665 | 0.970 |
| dense | 0.450 | 0.830 | 0.917 | 0.998 | 0.000 | 0.690 | 0.840 | 0.995 |
| hybrid | 0.455 | 0.807 | 0.897 | 0.995 | 0.000 | 0.630 | 0.795 | 0.990 |
| hybrid_rerank | 0.472 | 0.805 | 0.868 | 0.983 | 0.000 | 0.620 | 0.735 | 0.965 |

## qasper (n=175)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.169 | 0.352 | 0.450 | 0.601 | 0.126 | 0.269 | 0.337 | 0.486 |
| dense | 0.208 | 0.397 | 0.482 | 0.641 | 0.131 | 0.309 | 0.366 | 0.537 |
| hybrid | 0.243 | 0.403 | 0.504 | 0.651 | 0.171 | 0.303 | 0.383 | 0.531 |
| hybrid_rerank | 0.278 | 0.461 | 0.588 | 0.743 | 0.189 | 0.343 | 0.446 | 0.629 |
