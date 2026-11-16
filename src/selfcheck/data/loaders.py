"""Normalise each raw dataset into the unified ``Example`` / ``Document`` schema."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import Document, Example, Step, doc_id_for

Corpus = dict[str, Document]


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _add_doc(corpus: Corpus, dataset: str, title: str, text: str) -> str:
    title, text = _clean(title), _clean(text)
    did = doc_id_for(dataset, title, text)
    corpus.setdefault(did, Document(did, dataset, title, text))
    return did


def _uniq(xs: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def _majority(values: list[Any]) -> Any:
    """Majority vote; ties broken by the first annotator."""
    counts = Counter(values)
    top = max(counts.values())
    return next(v for v in values if counts[v] == top)


# --------------------------------------------------------------------------- MuSiQue
def load_musique(raw_dir: Path) -> tuple[list[Example], Corpus]:
    """Multi-hop (2-4 hops) with gold decomposition and per-hop supporting paragraph.

    The "full" version pairs every answerable question with an unanswerable twin
    whose key supporting paragraph was removed -> tests abstention.
    """
    corpus: Corpus = {}
    examples: list[Example] = []
    with open(raw_dir / "musique_full_v1.0_dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            idx2doc = {
                p["idx"]: _add_doc(corpus, "musique", p["title"], p["paragraph_text"])
                for p in r["paragraphs"]
            }
            gold = [idx2doc[p["idx"]] for p in r["paragraphs"] if p["is_supporting"]]
            steps = [
                Step(
                    sub_question=s["question"],
                    answer=s["answer"],
                    gold_doc_ids=[idx2doc[s["paragraph_support_idx"]]]
                    if s.get("paragraph_support_idx") is not None
                    else [],
                )
                for s in r["question_decomposition"]
            ]
            answerable = bool(r["answerable"])
            examples.append(
                Example(
                    qid=f"musique:{r['id']}",
                    dataset="musique",
                    question=r["question"],
                    answers=_uniq([r["answer"], *r.get("answer_aliases", [])]) if answerable else [],
                    answer_type="span" if answerable else "none",
                    answerable=answerable,
                    ambiguous=False,
                    gold_doc_ids=gold,
                    scope_doc_ids=list(idx2doc.values()),
                    steps=steps,
                    meta={
                        "hops": len(steps),
                        "composition": r["id"].split("__")[0],  # e.g. 2hop, 3hop1, 4hop3
                        "original_answer": r["answer"],
                    },
                )
            )
    return examples, corpus


# --------------------------------------------------------------------------- HotpotQA
def load_hotpotqa(raw_dir: Path) -> tuple[list[Example], Corpus]:
    """Bridge / comparison 2-hop questions; 2 gold + 8 distractor paragraphs each,
    with sentence-level supporting facts."""
    corpus: Corpus = {}
    examples: list[Example] = []
    df = pd.read_parquet(raw_dir / "hotpot_dist_val.parquet")
    for r in df.itertuples(index=False):
        title2doc: dict[str, str] = {}
        for title, sents in zip(r.context["title"], r.context["sentences"]):
            title2doc[title] = _add_doc(corpus, "hotpotqa", title, " ".join(sents))
        sf_titles = list(r.supporting_facts["title"])
        sf_ids = [int(i) for i in r.supporting_facts["sent_id"]]
        gold = _uniq(title2doc[t] for t in sf_titles if t in title2doc)
        ans = r.answer.strip()
        examples.append(
            Example(
                qid=f"hotpotqa:{r.id}",
                dataset="hotpotqa",
                question=r.question,
                answers=[ans],
                answer_type="yesno" if ans.lower() in ("yes", "no") else "span",
                answerable=True,
                ambiguous=False,
                gold_doc_ids=gold,
                scope_doc_ids=list(title2doc.values()),
                meta={
                    "hotpot_type": r.type,  # bridge | comparison
                    "hotpot_level": r.level,  # easy | medium | hard
                    "supporting_sentences": [[t, i] for t, i in zip(sf_titles, sf_ids)],
                },
            )
        )
    return examples, corpus


# --------------------------------------------------------------------------- Qasper
def _qasper_annotation(a: dict[str, Any]) -> tuple[str, str | None]:
    if a["unanswerable"]:
        return "none", None
    if a["yes_no"] is not None:
        return "yesno", "Yes" if a["yes_no"] else "No"
    if a["extractive_spans"]:
        return "span", ", ".join(s.strip() for s in a["extractive_spans"])
    if (a.get("free_form_answer") or "").strip():
        return "free", a["free_form_answer"].strip()
    return "none", None


def load_qasper(raw_dir: Path) -> tuple[list[Example], Corpus]:
    """QA over full NLP research papers. Questions were written from title+abstract
    only, so many are genuinely unanswerable from the paper -> tests abstention on
    long, technical documents. Retrieval scope = the paper's own paragraphs."""
    corpus: Corpus = {}
    examples: list[Example] = []
    for split, fname in (("dev", "qasper-dev-v0.3.json"), ("test", "qasper-test-v0.3.json")):
        papers = json.loads((raw_dir / "qasper" / fname).read_text())
        for pid, p in papers.items():
            text2doc: dict[str, str] = {}
            scope: list[str] = []
            title = p["title"]
            if p.get("abstract"):
                did = _add_doc(corpus, "qasper", f"{title} — Abstract", p["abstract"])
                text2doc[_clean(p["abstract"])] = did
                scope.append(did)
            for sec in p["full_text"]:
                sec_name = (sec.get("section_name") or "").split(":::")[-1].strip() or "Body"
                for para in sec["paragraphs"]:
                    if not para or not para.strip():
                        continue
                    did = _add_doc(corpus, "qasper", f"{title} — {sec_name}", para)
                    text2doc.setdefault(_clean(para), did)
                    scope.append(did)
            scope = _uniq(scope)

            for q in p["qas"]:
                anns = [_qasper_annotation(a["answer"]) for a in q["answers"]]
                if not anns:
                    continue
                answerable = _majority([t != "none" for t, _ in anns])
                answered = [(t, a) for t, a in anns if t != "none"]
                gold, n_float, n_unmatched = [], 0, 0
                for a in q["answers"]:
                    if a["answer"]["unanswerable"]:
                        continue
                    for ev in a["answer"]["evidence"]:
                        if ev.startswith("FLOAT SELECTED"):
                            n_float += 1  # evidence is a figure/table caption -> not retrievable text
                        elif _clean(ev) in text2doc:
                            gold.append(text2doc[_clean(ev)])
                        else:
                            n_unmatched += 1
                if answerable and not answered:  # cannot happen with majority, kept for safety
                    answerable = False
                examples.append(
                    Example(
                        qid=f"qasper:{q['question_id']}",
                        dataset="qasper",
                        question=q["question"],
                        answers=_uniq(a for _, a in answered) if answerable else [],
                        answer_type=_majority([t for t, _ in answered]) if answerable else "none",
                        answerable=answerable,
                        ambiguous=False,
                        gold_doc_ids=_uniq(gold) if answerable else [],
                        scope_doc_ids=scope,
                        meta={
                            "paper_id": pid,
                            "paper_title": title,
                            "source_split": split,  # dev papers -> our dev, test papers -> our test
                            "n_annotators": len(anns),
                            "annotator_answerable_votes": sum(t != "none" for t, _ in anns),
                            "float_evidence": n_float,
                            "unmatched_evidence": n_unmatched,
                        },
                    )
                )
    return examples, corpus


# --------------------------------------------------------------------------- AmbigQA
def _arr(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, np.ndarray)) else []


def load_ambigqa(raw_dir: Path) -> tuple[list[Example], Corpus]:
    """Open-domain questions labelled as unambiguous (single answer) or ambiguous
    (several plausible interpretations, each with its own disambiguated
    question + answer). Used to test the "ask for clarification" path.
    Passages = the search-result snippets annotators saw (no gold evidence labels)."""
    corpus: Corpus = {}
    examples: list[Example] = []
    df = pd.read_parquet(raw_dir / "ambig_full_val.parquet")
    for r in df.itertuples(index=False):
        scope: list[str] = []
        uq = r.used_queries
        for res in _arr(uq["results"]):
            for t, s in zip(_arr(res["title"]), _arr(res["snippet"])):
                if _clean(s):
                    scope.append(_add_doc(corpus, "ambigqa", t, s))
        scope = _uniq(scope)
        if not scope:
            continue
        ann = r.annotations
        types = list(ann["type"])
        ambiguous = _majority([t == "multipleQAs" for t in types])
        single = [a for t, a in zip(types, _arr(ann["answer"])) if t == "singleAnswer"]
        answers = _uniq(x for a in single for x in _arr(a)) or _uniq(_arr(r.nq_answer))
        disamb: list[dict[str, Any]] = []
        for t, qa in zip(types, _arr(ann["qaPairs"])):
            if t == "multipleQAs":
                disamb = [
                    {"question": dq, "answers": _uniq(_arr(da))}
                    for dq, da in zip(_arr(qa["question"]), _arr(qa["answer"]))
                ]
                break
        if not ambiguous and not answers:
            continue
        examples.append(
            Example(
                qid=f"ambigqa:{r.id}",
                dataset="ambigqa",
                question=r.question,
                answers=answers,
                answer_type="span",
                answerable=True,
                ambiguous=bool(ambiguous),
                gold_doc_ids=[],  # AmbigQA has no passage-level evidence labels
                scope_doc_ids=scope,
                disambiguations=disamb,
                meta={"annotation_types": types, "nq_answer": _uniq(_arr(r.nq_answer))},
            )
        )
    return examples, corpus


LOADERS = {
    "musique": load_musique,
    "hotpotqa": load_hotpotqa,
    "qasper": load_qasper,
    "ambigqa": load_ambigqa,
}
