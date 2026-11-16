"""Run the agent over a dataset split and log full traces (resumable).

    python scripts/run_agent.py --dataset musique --split dev --limit 20 --run baseline

Writes runs/<run>/<dataset>_<split>.jsonl, one trace per line. Questions already
in the file are skipped, so a run interrupted by the daily API quota simply
continues the next time it is started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.agent.agent import AgentConfig, DocQAAgent  # noqa: E402
from selfcheck.eval.metrics import score_final  # noqa: E402
from selfcheck.llm import DailyQuotaExceeded, LLMClient  # noqa: E402
from selfcheck.retrieval.index import CorpusIndex  # noqa: E402

GENERATOR = "openai/gpt-oss-20b"


def compact(trace: dict) -> dict:
    """Drop passage text from logs (recoverable from corpus.jsonl via chunk_id) to keep runs small."""
    for s in trace["steps"]:
        for r in s.get("retrieved", []):
            r.pop("text", None)
    return trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--model", default=GENERATOR)
    args = ap.parse_args()

    exs = [json.loads(l) for l in open(ROOT / "data/processed" / args.dataset / f"{args.split}.jsonl")]
    if args.limit:
        exs = exs[: args.limit]
    out_path = ROOT / "runs" / args.run / f"{args.dataset}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in open(out_path)} if out_path.exists() else set()

    index = CorpusIndex(ROOT / "data/processed", args.dataset, cache_dir=ROOT / ".cache/emb")
    agent = DocQAAgent(LLMClient(args.model, cache_path=ROOT / ".cache/llm.sqlite"), index, AgentConfig())

    todo = [e for e in exs if e["qid"] not in done]
    print(f"{len(done)} done, {len(todo)} to run")
    quota_hit = False
    with open(out_path, "a") as f:
        for i, e in enumerate(todo):
            try:
                t = agent.run(e["qid"], e["question"], e["scope_doc_ids"])
            except DailyQuotaExceeded as err:
                print(f"\nDaily quota reached after {i} questions; re-run later to resume.\n{err}")
                quota_hit = True
                break
            rec = compact(t.to_dict()) | {"dataset": args.dataset, "split": args.split, "model": args.model,
                                 "score": score_final(t.final_answer, e)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] {e['qid'][:40]:40s} steps={len(t.steps)} "
                  f"pred={t.final_answer[:40]!r} gold={(e['answers'] or ['<abstain>'])[0][:30]!r} "
                  f"f1={rec['score']['f1']:.2f}{' ERR ' + t.error if t.error else ''}")

    rows = [json.loads(l) for l in open(out_path)]
    if rows:
        print(f"\n{args.dataset}/{args.split}: n={len(rows)} "
              f"EM={mean(r['score']['em'] for r in rows):.3f} F1={mean(r['score']['f1'] for r in rows):.3f} "
              f"abstain={mean(r['abstained'] for r in rows):.2f} calls/q={mean(r['n_calls'] for r in rows):.1f} "
              f"tokens/q={mean(r['tokens'] for r in rows):.0f}")
    if quota_hit:
        sys.exit(3)


if __name__ == "__main__":
    main()
