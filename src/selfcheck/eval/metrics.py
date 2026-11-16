"""Automatic answer scoring (SQuAD-style normalisation, exact match, token F1).

These are the cheap first-pass labels. Paraphrased answers that EM/F1 miss are
graded by the LLM judge in the labelling stage.
"""

from __future__ import annotations

import re
import string
from collections import Counter

ABSTAIN = {"unanswerable", "unknown", "cannot be determined", "not enough information", ""}


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    pt = normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize(g).split()
        common = Counter(pt) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(pt), same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def is_abstention(pred: str) -> bool:
    return normalize(pred) in ABSTAIN


def contains_gold(pred: str, golds: list[str]) -> float:
    p = f" {normalize(pred)} "
    return float(any(f" {normalize(g)} " in p for g in golds if normalize(g)))


def score_final(pred: str, example: dict) -> dict:
    """Behaviour-aware score: abstaining is correct iff the question is unanswerable."""
    abst = is_abstention(pred)
    if not example["answerable"]:
        return {"em": float(abst), "f1": float(abst), "contains": float(abst), "abstained": abst}
    if abst:
        return {"em": 0.0, "f1": 0.0, "contains": 0.0, "abstained": True}
    golds = example["answers"]
    if example.get("ambiguous"):
        golds = golds + [a for d in example.get("disambiguations", []) for a in d["answers"]]
    return {"em": exact_match(pred, golds), "f1": token_f1(pred, golds),
            "contains": contains_gold(pred, golds), "abstained": False}
