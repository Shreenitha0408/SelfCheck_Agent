"""Measure retrieval quality on the dev split (gold evidence recall).

Two views:
* question-level: query = full question; how many gold passages are in the top-k?
* step-level (MuSiQue): query = gold sub-question with "#n" replaced by the gold
  answer of hop n; is that hop's gold passage in the top-k? This is the
  retrieval the agent actually needs at each step.

    python scripts/eval_retrieval.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.retrieval.index import CorpusIndex  # noqa: E402

KS = (1, 3, 5, 10)
METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")


def resolve(sub_q: str, answers: list[str]) -> str:
    return re.sub(r"#(\d+)", lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers) else m.group(0), sub_q)


def recall(ret, gold: list[str], k: int) -> tuple[float, float]:
    got = {r.doc_id for r in ret[:k]}
    hit = sum(g in got for g in gold)
    return hit / len(gold), float(hit == len(gold))


def main() -> None:
    proc = ROOT / "data/processed"
    results: dict = {}
    for ds in ("musique", "hotpotqa", "qasper"):
        t0 = time.time()
        index = CorpusIndex(proc, ds, cache_dir=ROOT / ".cache/emb")
        print(f"[{ds}] {len(index.chunks)} chunks indexed in {time.time() - t0:.0f}s")
        exs = [json.loads(l) for l in open(proc / ds / "dev.jsonl")]
        exs = [e for e in exs if e["answerable"] and e["gold_doc_ids"]]
        results[ds] = {}
        for m in METHODS:
            rec = {k: [] for k in KS}
            full = {k: [] for k in KS}
            step = {k: [] for k in KS}
            for e in exs:
                ret = index.search(e["question"], e["scope_doc_ids"], k=max(KS), method=m)
                for k in KS:
                    r, f = recall(ret, e["gold_doc_ids"], k)
                    rec[k].append(r)
                    full[k].append(f)
                gold_answers = [s["answer"] for s in e["steps"]]
                for s in e["steps"]:
                    if not s["gold_doc_ids"]:
                        continue
                    ret_s = index.search(resolve(s["sub_question"], gold_answers), e["scope_doc_ids"], k=max(KS), method=m)
                    for k in KS:
                        step[k].append(recall(ret_s, s["gold_doc_ids"], k)[0])
            results[ds][m] = {
                "n": len(exs),
                **{f"recall@{k}": round(mean(rec[k]), 3) for k in KS},
                **{f"all_gold@{k}": round(mean(full[k]), 3) for k in KS},
                **({f"step_recall@{k}": round(mean(step[k]), 3) for k in KS} if step[1] else {}),
            }
            print(ds, m, results[ds][m])

    out = ROOT / "reports"
    (out / "retrieval.json").write_text(json.dumps(results, indent=1))
    lines = ["# Retrieval evaluation (dev split, answerable questions)", "",
             "Scoped retrieval over each question's own passage pool. Chunks: ≤180 words, 40-word overlap.",
             "`recall@k` = share of gold passages in top-k; `all_gold@k` = all gold passages found;",
             "`step_recall@k` = per-hop gold passage found when querying with the gold sub-question (MuSiQue).", ""]
    for ds, ms in results.items():
        cols = [c for c in next(iter(ms.values())) if c != "n"]
        lines += [f"## {ds} (n={next(iter(ms.values()))['n']})", "",
                  "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for m, r in ms.items():
            lines.append(f"| {m} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        lines.append("")
    (out / "retrieval.md").write_text("\n".join(lines))
    print("wrote reports/retrieval.md")


if __name__ == "__main__":
    main()
