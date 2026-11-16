"""Attach correctness labels to agent traces (rules + batched LLM judge). Resumable.

    python scripts/label_runs.py --run baseline
    python scripts/label_runs.py --run baseline --audit 200   # also export a human-audit sheet

Writes runs/<run>/labels/<dataset>_<split>.jsonl:
    {qid, final_correct, final_source, steps: [{label, source}, ...]}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.eval.judge import Judge, reference_facts  # noqa: E402
from selfcheck.llm import DailyQuotaExceeded, LLMClient  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--audit", type=int, default=0, help="export N items for human labelling")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    judge = Judge(LLMClient("openai/gpt-oss-120b", cache_path=ROOT / ".cache/llm.sqlite"))
    all_labeled: list[tuple[dict, dict, dict]] = []
    quota = False

    for trace_file in sorted(run_dir.glob("*_*.jsonl")):
        ds, split = trace_file.stem.split("_", 1)
        examples = {e["qid"]: e for e in load_jsonl(ROOT / "data/processed" / ds / f"{split}.jsonl")}
        corpus = {d["doc_id"]: d for d in load_jsonl(ROOT / "data/processed" / ds / "corpus.jsonl")}
        out_path = run_dir / "labels" / trace_file.name
        out_path.parent.mkdir(exist_ok=True)
        done = {r["qid"]: r for r in load_jsonl(out_path)}
        traces = [t for t in load_jsonl(trace_file) if t["qid"] in examples]
        todo = [t for t in traces if t["qid"] not in done and not t.get("error")]

        if todo and not quota:
            try:
                finals = judge.label_finals([
                    {"key": t["qid"], "question": t["question"], "pred": t["final_answer"],
                     "golds": examples[t["qid"]]["answers"]
                     + [a for d in examples[t["qid"]].get("disambiguations", []) for a in d["answers"]],
                     "answerable": examples[t["qid"]]["answerable"]} for t in todo])
                steps = judge.label_steps([
                    {"key": f"{t['qid']}|{i}", "sub_question": s["sub_question"], "answer": s["answer"],
                     "found": s["found"], "reference": reference_facts(examples[t["qid"]], corpus)}
                    for t in todo for i, s in enumerate(t["steps"])])
                with open(out_path, "a") as f:
                    for t in todo:
                        rec = {"qid": t["qid"], "final_correct": finals[t["qid"]]["correct"],
                               "final_source": finals[t["qid"]]["source"],
                               "steps": [steps[f"{t['qid']}|{i}"] for i in range(len(t["steps"]))]}
                        f.write(json.dumps(rec) + "\n")
                        done[t["qid"]] = rec
            except DailyQuotaExceeded as e:
                print(f"judge quota reached: {e}")
                quota = True

        lab = [done[t["qid"]] for t in traces if t["qid"] in done]
        if lab:
            acc = sum(bool(r["final_correct"]) for r in lab) / len(lab)
            sl = [s["label"] for r in lab for s in r["steps"]]
            print(f"{ds}/{split}: labeled {len(lab)}/{len(traces)}  final acc={acc:.3f}  "
                  f"steps: correct={sl.count('correct')} incorrect={sl.count('incorrect')} unknown={sl.count('unknown')}")
        all_labeled += [(t, done[t["qid"]], examples[t["qid"]]) for t in traces if t["qid"] in done]

    if args.audit and all_labeled:
        rng = random.Random(0)
        rows = []
        finals = [(t, l, e) for t, l, e in all_labeled if l["final_source"] == "judge"]
        for t, l, e in rng.sample(finals, min(args.audit // 2, len(finals))):
            rows.append({"kind": "final", "id": t["qid"], "question": t["question"],
                         "item": t["final_answer"], "reference": " | ".join(e["answers"][:5]),
                         "judge": l["final_correct"], "human_correct (1/0)": ""})
        steps = [(t, i, s, l["steps"][i], e) for t, l, e in all_labeled for i, s in enumerate(t["steps"])
                 if l["steps"][i]["label"] in ("correct", "incorrect")]
        for t, i, s, sl, e in rng.sample(steps, min(args.audit - len(rows), len(steps))):
            rows.append({"kind": "step", "id": f"{t['qid']}|{i}", "question": s["sub_question"],
                         "item": s["answer"] if s["found"] else "NOT FOUND",
                         "reference": " ; ".join(f"{h['sub_question']} -> {h['answer']}" for h in e.get("steps", []))
                         or " | ".join(e["answers"][:3]),
                         "judge": sl["label"] == "correct", "human_correct (1/0)": ""})
        rng.shuffle(rows)
        audit = ROOT / "reports" / "human_audit" / f"{args.run}_audit.csv"
        audit.parent.mkdir(parents=True, exist_ok=True)
        with open(audit, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {audit} ({len(rows)} items) — fill the last column, judge column can be hidden")


if __name__ == "__main__":
    main()
