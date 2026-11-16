# Retrieval evaluation (dev split, answerable questions)

Scoped retrieval over each question's own passage pool. Chunks: ≤180 words, 40-word overlap.
`recall@k` = share of gold passages in top-k; `all_gold@k` = all gold passages found;
`step_recall@k` = per-hop gold passage found when querying with the gold sub-question (MuSiQue).

## musique (n=50)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 | step_recall@1 | step_recall@3 | step_recall@5 | step_recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.255 | 0.443 | 0.562 | 0.692 | 0.000 | 0.120 | 0.200 | 0.360 | 0.712 | 0.932 | 0.977 | 0.985 |
| dense | 0.358 | 0.595 | 0.733 | 0.862 | 0.000 | 0.220 | 0.440 | 0.640 | 0.803 | 0.939 | 0.992 | 0.992 |
| hybrid | 0.320 | 0.547 | 0.660 | 0.768 | 0.000 | 0.180 | 0.340 | 0.520 | 0.788 | 0.977 | 0.985 | 0.992 |
| hybrid_rerank | 0.383 | 0.585 | 0.657 | 0.792 | 0.000 | 0.220 | 0.340 | 0.540 | 0.886 | 0.985 | 0.992 | 0.992 |

## hotpotqa (n=100)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.390 | 0.695 | 0.800 | 0.945 | 0.000 | 0.430 | 0.620 | 0.890 |
| dense | 0.430 | 0.830 | 0.910 | 0.995 | 0.000 | 0.680 | 0.820 | 0.990 |
| hybrid | 0.445 | 0.760 | 0.860 | 0.995 | 0.000 | 0.530 | 0.730 | 0.990 |
| hybrid_rerank | 0.475 | 0.750 | 0.855 | 0.960 | 0.000 | 0.520 | 0.710 | 0.920 |

## qasper (n=85)

| method | recall@1 | recall@3 | recall@5 | recall@10 | all_gold@1 | all_gold@3 | all_gold@5 | all_gold@10 |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.175 | 0.355 | 0.487 | 0.663 | 0.106 | 0.259 | 0.353 | 0.529 |
| dense | 0.218 | 0.452 | 0.577 | 0.691 | 0.153 | 0.353 | 0.459 | 0.565 |
| hybrid | 0.232 | 0.463 | 0.569 | 0.737 | 0.165 | 0.341 | 0.424 | 0.600 |
| hybrid_rerank | 0.312 | 0.507 | 0.639 | 0.806 | 0.212 | 0.376 | 0.482 | 0.682 |
