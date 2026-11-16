"""Deterministic stratified dev/test sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .schema import Example

STRATUM_FIELDS = {
    "answerable": lambda e: e.answerable,
    "ambiguous": lambda e: e.ambiguous,
    "answer_type": lambda e: e.answer_type,
}


def stratum_key(e: Example, fields: list[str]) -> tuple[Any, ...]:
    return tuple(STRATUM_FIELDS[f](e) if f in STRATUM_FIELDS else e.meta[f] for f in fields)


def _allocate(sizes: dict[Any, int], n: int) -> dict[Any, int]:
    """Proportional allocation with largest-remainder rounding, capped by stratum size."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], int(r)) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - int(raw[k]), str(k)), reverse=True):
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    return alloc


def stratified_sample(pool: list[Example], n: int, fields: list[str], seed: int) -> list[Example]:
    groups: dict[Any, list[Example]] = defaultdict(list)
    for e in sorted(pool, key=lambda e: e.qid):
        groups[stratum_key(e, fields)].append(e)
    alloc = _allocate({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    out: list[Example] = []
    for k in sorted(groups, key=str):
        out.extend(rng.sample(groups[k], alloc[k]))
    rng.shuffle(out)
    return out


def make_splits(
    dataset: str, examples: list[Example], n_dev: int, n_test: int, fields: list[str], seed: int
) -> dict[str, list[Example]]:
    if dataset == "qasper":
        # Paper-level split: dev questions from dev papers, test from test papers
        # -> no paper appears in both.
        dev_pool = [e for e in examples if e.meta["source_split"] == "dev"]
        test_pool = [e for e in examples if e.meta["source_split"] == "test"]
        dev = stratified_sample(dev_pool, n_dev, fields, seed)
        test = stratified_sample(test_pool, n_test, fields, seed + 1)
    else:
        test = stratified_sample(examples, n_test, fields, seed + 1)
        taken = {e.qid for e in test}
        dev = stratified_sample([e for e in examples if e.qid not in taken], n_dev, fields, seed)
    assert not {e.qid for e in dev} & {e.qid for e in test}
    return {"dev": dev, "test": test}
