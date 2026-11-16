"""Fit the confidence combiner on dev, evaluate the verifier on held-out test.

    python scripts/evaluate_verifier.py --run baseline

Needs runs/<run>/{*.jsonl, labels/, signals/} for dev and test.
Reports (reports/verifier_eval.{md,json}):
  * step level   : AUROC / ECE per signal and combined; catch rate (TP) vs false-flag rate (FP)
  * question level: accuracy without verification vs with (flag -> abstain), coverage, risk-coverage
  * ablations    : each signal group alone and removed
Threshold tau is chosen on dev (false-flag rate <= 10%) and frozen before touching test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.eval.metrics import is_abstention  # noqa: E402
from selfcheck.verify.confidence import (  # noqa: E402
    ConfidenceModel, auroc, bootstrap_ci, choose_tau, dump, ece, feature_vector, flag_metrics, risk_coverage)
from selfcheck.verify.signals import FEATURE_GROUPS, NUMERIC_FEATURES  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def collect(run_dir: Path, split: str) -> tuple[list[dict], list[dict]]:
    """Return (step rows, question rows) for one split, joined across traces/labels/signals."""
    steps, questions = [], []
    for tf in sorted(run_dir.glob(f"*_{split}.jsonl")):
        ds = tf.stem.split("_")[0]
        ex = {e["qid"]: e for e in load_jsonl(ROOT / "data/processed" / ds / f"{split}.jsonl")}
        lab = {r["qid"]: r for r in load_jsonl(run_dir / "labels" / tf.name)}
        sig = {r["qid"]: r for r in load_jsonl(run_dir / "signals" / tf.name)}
        for t in load_jsonl(tf):
            q = t["qid"]
            if q not in lab or q not in sig or lab[q]["final_correct"] is None or not t["steps"]:
                continue
            srows = []
            for i, (s, sl) in enumerate(zip(sig[q]["steps"], lab[q]["steps"])):
                row = {"dataset": ds, "qid": q, "i": i, "sig": s, "label": sl.get("label")}
                srows.append(row)
                if row["label"] in ("correct", "incorrect"):
                    steps.append(row)
            questions.append({"dataset": ds, "qid": q, "steps": srows, "correct": bool(lab[q]["final_correct"]),
                              "answerable": ex[q]["answerable"], "ambiguous": ex[q]["ambiguous"],
                              "abstained": is_abstention(t["final_answer"]), "n_calls": t["n_calls"],
                              "tokens": t["tokens"]})
    return steps, questions


def xy(rows: list[dict], feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([feature_vector(r["sig"], feats) for r in rows], dtype=float)
    y = np.array([r["label"] == "correct" for r in rows], dtype=int)
    return X, y


def question_conf(model: ConfidenceModel, q: dict) -> float:
    """A final answer is only as trustworthy as its weakest step."""
    return min(model.score_step(s["sig"]) for s in q["steps"])


def abstain_accuracy(qs: list[dict], conf: np.ndarray, tau: float) -> float:
    """Flagged answers are replaced by an abstention; abstaining is right iff unanswerable."""
    ok = []
    for q, c in zip(qs, conf):
        if c < tau and not q["abstained"]:
            ok.append(not q["answerable"])
        else:
            ok.append(q["correct"])
    return float(np.mean(ok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--max-false-flag", type=float, default=0.10)
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run

    dev_s, dev_q = collect(run_dir, "dev")
    test_s, test_q = collect(run_dir, "test")
    print(f"dev: {len(dev_s)} labelled steps / {len(dev_q)} questions; test: {len(test_s)} / {len(test_q)}")
    if not dev_s or not test_s:
        sys.exit("need labelled + signalled dev and test runs")

    Xd, yd = xy(dev_s, NUMERIC_FEATURES)
    Xt, yt = xy(test_s, NUMERIC_FEATURES)
    model = ConfidenceModel.fit(Xd, yd, NUMERIC_FEATURES)
    model.save(ROOT / "models" / "confidence.pkl")
    ct = model.predict(Xt)

    res: dict = {"n": {"dev_steps": len(dev_s), "test_steps": len(test_s), "dev_q": len(dev_q), "test_q": len(test_q)},
                 "coefficients": model.coefficients()}

    # ---------------- step level: each raw signal vs combined
    single = {"critic_conf": "critic_conf", "nli_entail_max": "nli_entail_max", "agree_f1": "agree_f1"}
    step_tbl = {}
    for name, f in single.items():
        s = Xt[:, NUMERIC_FEATURES.index(f)]
        step_tbl[name] = {"auroc": auroc(s, yt)}
    step_tbl["combined"] = {"auroc": auroc(ct, yt), "auroc_ci": bootstrap_ci(auroc, ct, yt),
                            "ece": ece(ct, yt)}
    tau_step = choose_tau(model.predict(Xd), yd, args.max_false_flag)
    step_tbl["combined"]["flags"] = flag_metrics(ct, yt, tau_step)
    res["step"] = step_tbl

    # ---------------- question level
    qd = np.array([question_conf(model, q) for q in dev_q])
    yqd = np.array([q["correct"] for q in dev_q], dtype=int)
    qt = np.array([question_conf(model, q) for q in test_q])
    yqt = np.array([q["correct"] for q in test_q], dtype=int)
    tau_q = choose_tau(qd, yqd, args.max_false_flag)
    fm = flag_metrics(qt, yqt, tau_q)
    fm["catch_rate_ci"] = bootstrap_ci(lambda c, y: flag_metrics(c, y, tau_q)["catch_rate"], qt, yqt)
    fm["false_flag_rate_ci"] = bootstrap_ci(lambda c, y: flag_metrics(c, y, tau_q)["false_flag_rate"], qt, yqt)
    cov, acc, aurc = risk_coverage(qt, yqt)
    res["question"] = {
        "auroc": auroc(qt, yqt), "auroc_ci": bootstrap_ci(auroc, qt, yqt), "ece": ece(qt, yqt),
        "flags": fm, "aurc": aurc,
        "accuracy_without_verifier": float(yqt.mean()),
        "accuracy_with_verifier_abstain": abstain_accuracy(test_q, qt, tau_q),
        "risk_coverage": {"coverage": cov, "accuracy": acc},
    }

    per_ds = defaultdict(dict)
    for ds in sorted({q["dataset"] for q in test_q}):
        idx = [i for i, q in enumerate(test_q) if q["dataset"] == ds]
        sub = [test_q[i] for i in idx]
        f = flag_metrics(qt[idx], yqt[idx], tau_q)
        per_ds[ds] = {"n": len(idx), "auroc": auroc(qt[idx], yqt[idx]),
                      "acc_without": f["accuracy_all"], "acc_unflagged": f["accuracy_unflagged"],
                      "coverage": f["coverage"], "catch_rate": f["catch_rate"],
                      "false_flag_rate": f["false_flag_rate"],
                      "acc_with_abstain": abstain_accuracy(sub, qt[idx], tau_q)}
    res["per_dataset"] = per_ds

    # ---------------- ablations (question-level AUROC)
    abl = {}
    for g, feats in FEATURE_GROUPS.items():
        for mode, fs in (("only", feats), ("without", [f for f in NUMERIC_FEATURES if f not in feats])):
            m = ConfidenceModel.fit(*xy(dev_s, fs), fs)
            qq = np.array([min(m.score_step(s["sig"]) for s in q["steps"]) for q in test_q])
            abl[f"{mode} {g}"] = {"question_auroc": auroc(qq, yqt), "step_auroc": auroc(m.predict(xy(test_s, fs)[0]), yt)}
    res["ablations"] = abl
    res["cost"] = {"baseline_calls_per_q": float(np.mean([q["n_calls"] for q in test_q])),
                   "baseline_tokens_per_q": float(np.mean([q["tokens"] for q in test_q])),
                   "verifier_calls_per_step": 2}
    dump(res, ROOT / "reports" / "verifier_eval.json")
    write_md(res)
    print((ROOT / "reports" / "verifier_eval.md").read_text())


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ci(c) -> str:
    return f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def write_md(r: dict) -> None:
    q, fq = r["question"], r["question"]["flags"]
    L = ["# Verifier evaluation (held-out test)", "",
         f"Confidence model fit on dev ({r['n']['dev_steps']} labelled steps); threshold chosen on dev "
         f"(false-flag rate ≤ 10%); evaluated once on test ({r['n']['test_q']} questions, {r['n']['test_steps']} steps).",
         "95% bootstrap intervals in brackets.", "",
         "## Headline (question level)", "",
         "| metric | value |", "|---|---|",
         f"| Wrong answers caught (true-positive flags) | {pct(fq['catch_rate'])}{ci(fq['catch_rate_ci'])} — {fq['TP']}/{fq['TP'] + fq['FN']} |",
         f"| Correct answers wrongly flagged (false positives) | {pct(fq['false_flag_rate'])}{ci(fq['false_flag_rate_ci'])} — {fq['FP']}/{fq['FP'] + fq['TN']} |",
         f"| Flag precision | {pct(fq['flag_precision'])} |",
         f"| Accuracy without verification | {pct(q['accuracy_without_verifier'])} |",
         f"| Accuracy on answers the verifier lets through | {pct(fq['accuracy_unflagged'])} at {pct(fq['coverage'])} coverage |",
         f"| Accuracy with verification (flag → abstain) | {pct(q['accuracy_with_verifier_abstain'])} |",
         f"| AUROC of confidence | {q['auroc']:.3f}{ci(q['auroc_ci']).replace('%', '')} |",
         f"| Calibration error (ECE) | {q['ece']:.3f} |", "",
         "## Per dataset", "", "| dataset | n | acc. without | acc. unflagged | coverage | caught | false flags | acc. flag→abstain | AUROC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ds, d in r["per_dataset"].items():
        L.append(f"| {ds} | {d['n']} | {pct(d['acc_without'])} | {pct(d['acc_unflagged'])} | {pct(d['coverage'])} | "
                 f"{pct(d['catch_rate'])} | {pct(d['false_flag_rate'])} | {pct(d['acc_with_abstain'])} | {d['auroc']:.3f} |")
    s = r["step"]
    L += ["", "## Step level (each verification signal alone vs combined)", "", "| signal | AUROC |", "|---|---|"]
    for k, v in s.items():
        L.append(f"| {k} | {v['auroc']:.3f} |")
    L += ["", "## Ablations (AUROC)", "", "| features | question | step |", "|---|---|---|"]
    for k, v in r["ablations"].items():
        L.append(f"| {k} | {v['question_auroc']:.3f} | {v['step_auroc']:.3f} |")
    L += ["", "## Combiner coefficients (standardised)", "", "| feature | weight |", "|---|---|"]
    L += [f"| {k} | {v:+.3f} |" for k, v in r["coefficients"].items()]
    (ROOT / "reports" / "verifier_eval.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
