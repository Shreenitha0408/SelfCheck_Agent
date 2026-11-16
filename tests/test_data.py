import json
from pathlib import Path

import pytest

from selfcheck.data.schema import Example
from selfcheck.data.splits import _allocate, make_splits, stratified_sample

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def _ex(i: int, answerable: bool) -> Example:
    return Example(
        qid=f"t:{i}", dataset="t", question="q?", answers=["a"] if answerable else [],
        answer_type="span" if answerable else "none", answerable=answerable, ambiguous=False,
        gold_doc_ids=[], scope_doc_ids=[], meta={"source_split": "dev"},
    )


def test_allocate_exact_and_proportional():
    alloc = _allocate({"a": 700, "b": 300}, 100)
    assert alloc == {"a": 70, "b": 30}
    assert sum(_allocate({"a": 5, "b": 3, "c": 1}, 7).values()) == 7


def test_sampling_deterministic_and_stratified():
    pool = [_ex(i, i % 4 != 0) for i in range(400)]  # 75% answerable
    s1 = stratified_sample(pool, 100, ["answerable"], seed=1)
    s2 = stratified_sample(list(reversed(pool)), 100, ["answerable"], seed=1)
    assert [e.qid for e in s1] == [e.qid for e in s2]
    assert sum(e.answerable for e in s1) == 75


def test_splits_disjoint():
    pool = [_ex(i, True) for i in range(300)]
    sp = make_splits("t", pool, 50, 100, ["answerable"], seed=0)
    assert len(sp["dev"]) == 50 and len(sp["test"]) == 100
    assert not {e.qid for e in sp["dev"]} & {e.qid for e in sp["test"]}


@pytest.mark.skipif(not PROC.exists(), reason="run scripts/prepare_data.py first")
@pytest.mark.parametrize("ds", ["musique", "hotpotqa", "qasper", "ambigqa"])
def test_processed_integrity(ds):
    corpus = {json.loads(l)["doc_id"] for l in open(PROC / ds / "corpus.jsonl")}
    for split in ("dev", "test"):
        for line in open(PROC / ds / f"{split}.jsonl"):
            e = Example.from_dict(json.loads(line))
            assert e.scope_doc_ids and set(e.scope_doc_ids) <= corpus
            assert set(e.gold_doc_ids) <= set(e.scope_doc_ids)
            for s in e.steps:
                assert set(s.gold_doc_ids) <= set(e.scope_doc_ids)
            if not e.answerable:
                assert e.answers == [] and e.answer_type == "none"
            if e.ambiguous:
                assert e.disambiguations
