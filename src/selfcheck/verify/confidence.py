"""Confidence combiner + evaluation metrics for the verifier.

The combiner is a logistic regression over the verification signals, fit on
the dev split only. Its output is a probability that a step is correct.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .signals import NUMERIC_FEATURES


def feature_vector(sig: dict[str, Any], features: list[str] = NUMERIC_FEATURES) -> list[float]:
    return [float(sig.get(f, 0.0) or 0.0) for f in features]


@dataclass
class ConfidenceModel:
    features: list[str]
    pipe: Any

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, features: list[str]) -> "ConfidenceModel":
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight=None))
        pipe.fit(X, y)
        return cls(features, pipe)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]

    def score_step(self, sig: dict[str, Any]) -> float:
        return float(self.predict(np.array([feature_vector(sig, self.features)]))[0])

    def coefficients(self) -> dict[str, float]:
        lr = self.pipe[-1]
        return {f: round(float(c), 3) for f, c in zip(self.features, lr.coef_[0])}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "ConfidenceModel":
        return pickle.loads(path.read_bytes())


# ------------------------------------------------------------------ metrics
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(roc_auc_score(correct, conf)) if len(set(correct.tolist())) == 2 else float("nan")


def flag_metrics(conf: np.ndarray, correct: np.ndarray, tau: float) -> dict[str, float]:
    """A flag = conf < tau. Positive class for flagging = the answer is WRONG."""
    flagged = conf < tau
    wrong = correct == 0
    tp = int((flagged & wrong).sum())
    fp = int((flagged & ~wrong).sum())
    fn = int((~flagged & wrong).sum())
    tn = int((~flagged & ~wrong).sum())
    return {
        "tau": float(tau), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "catch_rate": tp / max(tp + fn, 1),          # share of wrong answers that get flagged (recall)
        "false_flag_rate": fp / max(fp + tn, 1),     # share of correct answers wrongly flagged
        "flag_precision": tp / max(tp + fp, 1),      # share of flags that were really wrong
        "flag_rate": float(flagged.mean()),
        "accuracy_all": float(correct.mean()),
        "accuracy_unflagged": float(correct[~flagged].mean()) if (~flagged).any() else float("nan"),
        "coverage": float((~flagged).mean()),
    }


def choose_tau(conf: np.ndarray, correct: np.ndarray, max_false_flag: float = 0.10) -> float:
    """Highest threshold whose false-flag rate on dev stays <= max_false_flag."""
    best = 0.0
    for tau in np.unique(np.concatenate([[0.0], conf])):
        if flag_metrics(conf, correct, tau)["false_flag_rate"] <= max_false_flag:
            best = max(best, float(tau))
    return best


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[list[float], list[float], float]:
    order = np.argsort(-conf)
    c = correct[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    aurc = float(np.mean(1 - acc))
    return cov.tolist(), acc.tolist(), aurc


def bootstrap_ci(fn, *arrays: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    size = len(arrays[0])
    for _ in range(n):
        idx = rng.integers(0, size, size)
        v = fn(*(a[idx] for a in arrays))
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
