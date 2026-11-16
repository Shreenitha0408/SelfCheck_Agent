"""Compute verification signals for every step of every trace in a run (resumable).

    python scripts/compute_signals.py --run baseline

Signals are computed *offline* on the baseline traces, so the verifier's
catch rate / false-flag rate can be measured on exactly the steps the
unverified agent produced. Output: runs/<run>/signals/<dataset>_<split>.jsonl
    {qid, steps: [{...features...}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.llm import DailyQuotaExceeded, LLMClient  # noqa: E402
from selfcheck.retrieval.index import chunk_document  # noqa: E402
from selfcheck.verify.nli import NLIChecker  # noqa: E402
from selfcheck.verify.signals import CRITIC_MODEL, INDEPENDENT_MODEL, StepVerifier  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--no-llm", action="store_true", help="only local signals (NLI + cheap features)")
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()

    cache = ROOT / ".cache/llm.sqlite"
    ver = StepVerifier(NLIChecker(), LLMClient(CRITIC_MODEL, cache_path=cache),
                       LLMClient(INDEPENDENT_MODEL, cache_path=cache))
    run_dir = ROOT / "runs" / args.run
    for trace_file in sorted(run_dir.glob("*_*.jsonl")):
        ds, split = trace_file.stem.split("_", 1)
        if args.datasets and ds not in args.datasets:
            continue
        chunks = {c.chunk_id: c for d in load_jsonl(ROOT / "data/processed" / ds / "corpus.jsonl")
                  for c in chunk_document(d)}
        out = run_dir / "signals" / trace_file.name
        out.parent.mkdir(exist_ok=True)
        done = {r["qid"] for r in load_jsonl(out)}
        todo = [t for t in load_jsonl(trace_file) if t["qid"] not in done and not t.get("error")]
        print(f"{ds}/{split}: {len(done)} done, {len(todo)} to do")
        with open(out, "a") as f:
            for n, t in enumerate(todo):
                try:
                    steps = []
                    for s in t["steps"]:
                        passages = [{"chunk_id": r["chunk_id"], "title": chunks[r["chunk_id"]].title,
                                     "text": chunks[r["chunk_id"]].text, "score": r["score"]}
                                    for r in s["retrieved"] if r["chunk_id"] in chunks]
                        steps.append(ver.signals(s["sub_question"], s["answer"], s["found"], passages,
                                                 s["cited_chunk_ids"], use_llm=not args.no_llm,
                                                 claim=s.get("claim", "")))
                except DailyQuotaExceeded as e:
                    print(f"quota reached after {n} questions: {e}")
                    return
                f.write(json.dumps({"qid": t["qid"], "steps": steps}) + "\n")
                f.flush()
                if (n + 1) % 20 == 0:
                    print(f"  {n + 1}/{len(todo)}")


if __name__ == "__main__":
    main()
